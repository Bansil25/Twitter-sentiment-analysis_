"""
ModelService — abstraction that loads both models and routes inference.
This is the "service layer" pattern from clean architecture:
  API routes → ModelService → concrete model implementations
Routes never import tensorflow/transformers directly.
"""

from __future__ import annotations

import asyncio
import functools
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from app.core.config import settings
from app.schemas.sentiment import ModelName, SentimentLabel

log = structlog.get_logger()

# Label mapping — must match training label encoding
LABEL_MAP = {0: SentimentLabel.NEGATIVE, 1: SentimentLabel.POSITIVE,
             2: SentimentLabel.NEUTRAL,  3: SentimentLabel.IRRELEVANT}


class ModelService:
    """
    Loads and manages both models.
    Inference is dispatched to a thread pool (run_in_executor) so it
    doesn't block the async event loop during CPU/GPU work.
    """

    def __init__(self):
        self._models: Dict[ModelName, Any] = {}
        self._tokenizers: Dict[ModelName, Any] = {}
        self._executor = None  # ThreadPoolExecutor set at load time

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def load(self) -> None:
        """Load all models at startup. Called from lifespan context manager."""
        import concurrent.futures
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(self._executor, self._load_distilbert),
            loop.run_in_executor(self._executor, self._load_bilstm),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for name, result in zip([ModelName.DISTILBERT, ModelName.BILSTM], results):
            if isinstance(result, Exception):
                log.warning(f"model.load.failed", model=name, error=str(result))

    async def unload(self) -> None:
        if self._executor:
            self._executor.shutdown(wait=False)

    async def reload(self, model_name: ModelName) -> None:
        loop = asyncio.get_running_loop()
        loader = self._load_distilbert if model_name == ModelName.DISTILBERT else self._load_bilstm
        await loop.run_in_executor(self._executor, loader)

    @property
    def loaded_models(self) -> List[str]:
        return [m.value for m in self._models]

    # ── Inference ──────────────────────────────────────────────────────────

    async def predict(
        self,
        text: str,
        model_name: ModelName,
        explain: bool = False,
    ) -> Dict[str, Any]:
        if model_name not in self._models:
            # Graceful fallback: try the other model
            fallback = ModelName.BILSTM if model_name == ModelName.DISTILBERT else ModelName.DISTILBERT
            if fallback in self._models:
                log.warning("model.fallback", requested=model_name, using=fallback)
                model_name = fallback
            else:
                raise RuntimeError("No models available for inference")

        loop = asyncio.get_running_loop()
        fn = functools.partial(self._sync_predict, text, model_name, explain)
        return await loop.run_in_executor(self._executor, fn)

    async def batch_predict(
        self,
        texts: List[str],
        model_name: ModelName,
    ) -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        fn = functools.partial(self._sync_batch_predict, texts, model_name)
        return await loop.run_in_executor(self._executor, fn)

    # ── Sync inference (runs in thread pool) ──────────────────────────────

    def _sync_predict(self, text: str, model_name: ModelName, explain: bool) -> Dict[str, Any]:
        if model_name == ModelName.DISTILBERT:
            return self._distilbert_predict([text], explain)[0]
        return self._bilstm_predict([text])[0]

    def _sync_batch_predict(self, texts: List[str], model_name: ModelName) -> List[Dict[str, Any]]:
        if model_name == ModelName.DISTILBERT:
            return self._distilbert_predict(texts, explain=False)
        return self._bilstm_predict(texts)

    # ── DistilBERT inference ───────────────────────────────────────────────

    def _distilbert_predict(self, texts: List[str], explain: bool) -> List[Dict[str, Any]]:
        import numpy as np
        import torch
        import torch.nn.functional as F

        tokenizer = self._tokenizers[ModelName.DISTILBERT]
        model = self._models[ModelName.DISTILBERT]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=settings.MAX_SEQUENCE_LENGTH,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).numpy()

        results = []
        for i, prob in enumerate(probs):
            label_idx = int(np.argmax(prob))
            sentiment = LABEL_MAP.get(label_idx, SentimentLabel.NEUTRAL)
            probabilities = {
                SentimentLabel.NEGATIVE: round(float(prob[0]), 4),
                SentimentLabel.POSITIVE: round(float(prob[1]), 4),
                SentimentLabel.NEUTRAL:  round(float(prob[2]), 4),
                SentimentLabel.IRRELEVANT: round(float(prob[3]), 4),
            }
            result = {
                "text": texts[i],
                "sentiment": sentiment,
                "confidence": round(float(prob[label_idx]), 4),
                "probabilities": probabilities,
            }
            if explain and i == 0:  # SHAP only for single predictions
                result["explanation"] = self._explain_distilbert(texts[i])
            results.append(result)
        return results

    # ── Bi-LSTM inference ─────────────────────────────────────────────────

    def _bilstm_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        import numpy as np
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        tokenizer = self._tokenizers[ModelName.BILSTM]
        model = self._models[ModelName.BILSTM]

        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=settings.MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
        probs_batch = model.predict(padded, verbose=0)

        results = []
        for i, prob in enumerate(probs_batch):
            label_idx = int(np.argmax(prob))
            sentiment = LABEL_MAP.get(label_idx, SentimentLabel.NEUTRAL)
            all_labels = [SentimentLabel.NEGATIVE, SentimentLabel.POSITIVE,
                          SentimentLabel.NEUTRAL, SentimentLabel.IRRELEVANT]
            probabilities = {label: round(float(prob[j]), 4) for j, label in enumerate(all_labels)}
            results.append({
                "text": texts[i],
                "sentiment": sentiment,
                "confidence": round(float(prob[label_idx]), 4),
                "probabilities": probabilities,
            })
        return results

    # ── SHAP explainability ────────────────────────────────────────────────

    def _explain_distilbert(self, text: str) -> List[Dict[str, Any]]:
        """Token-level SHAP values for DistilBERT predictions."""
        try:
            import shap
            import numpy as np

            tokenizer = self._tokenizers[ModelName.DISTILBERT]
            model = self._models[ModelName.DISTILBERT]

            def predict_fn(texts):
                import torch, torch.nn.functional as F
                inputs = tokenizer(list(texts), return_tensors="pt", truncation=True, padding=True, max_length=128)
                with torch.no_grad():
                    logits = model(**inputs).logits
                return F.softmax(logits, dim=-1).numpy()

            explainer = shap.Explainer(predict_fn, tokenizer)
            shap_values = explainer([text])
            tokens = tokenizer.tokenize(text)
            scores = shap_values.values[0][:, 0]  # scores for class 0

            return [{"token": t, "score": round(float(s), 4)}
                    for t, s in zip(tokens[:30], scores[:30])]
        except Exception as e:
            log.warning("explain.failed", error=str(e))
            return []

    # ── Model loaders (called in thread pool) ─────────────────────────────

    def _load_distilbert(self) -> None:
        model_path = Path(settings.DISTILBERT_MODEL_PATH)
        if not model_path.exists():
            log.warning("distilbert.not_found", path=str(model_path))
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            log.info("distilbert.loading", path=str(model_path))
            self._tokenizers[ModelName.DISTILBERT] = AutoTokenizer.from_pretrained(str(model_path))
            self._models[ModelName.DISTILBERT] = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            self._models[ModelName.DISTILBERT].eval()
            log.info("distilbert.loaded")
        except Exception as e:
            log.error("distilbert.load_error", error=str(e))
            raise

    def _load_bilstm(self) -> None:
        model_path = Path(settings.BILSTM_MODEL_PATH)
        tok_path = Path(settings.BILSTM_TOKENIZER_PATH)
        if not model_path.exists():
            log.warning("bilstm.not_found", path=str(model_path))
            return
        try:
            import tensorflow as tf
            log.info("bilstm.loading", path=str(model_path))
            self._models[ModelName.BILSTM] = tf.keras.models.load_model(str(model_path))
            with open(tok_path, "rb") as f:
                self._tokenizers[ModelName.BILSTM] = pickle.load(f)
            log.info("bilstm.loaded")
        except Exception as e:
            log.error("bilstm.load_error", error=str(e))
            raise
