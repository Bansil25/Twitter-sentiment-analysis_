"""
Extended ModelService with Phase 3 additions:
  - SHAP explainability via SHAPExplainer
  - Drift detection via DriftDetector
  - Prometheus counters for each prediction

This replaces app/services/model_service.py
"""

from __future__ import annotations

import asyncio
import functools
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from prometheus_client import Counter, Histogram

from app.core.config import settings
from app.schemas.sentiment import ModelName, SentimentLabel

log = structlog.get_logger()

LABEL_MAP = {
    0: SentimentLabel.NEGATIVE,
    1: SentimentLabel.POSITIVE,
    2: SentimentLabel.NEUTRAL,
    3: SentimentLabel.IRRELEVANT,
}

# ── Prometheus counters ────────────────────────────────────────────────────
PREDICTION_COUNTER = Counter(
    "sentimentai_predictions_total",
    "Total predictions",
    ["model", "sentiment"],
)
PREDICTION_LATENCY = Histogram(
    "sentimentai_inference_latency_seconds",
    "Per-prediction latency",
    ["model"],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0],
)
CONFIDENCE_HISTOGRAM = Histogram(
    "sentimentai_prediction_confidence",
    "Distribution of model confidence scores",
    ["model"],
    buckets=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0],
)


class ModelService:
    """
    Central service for ML model loading, inference, explainability, and drift monitoring.
    Single instance shared across all FastAPI workers via app.state.
    """

    def __init__(self):
        self._models: Dict[ModelName, Any]     = {}
        self._tokenizers: Dict[ModelName, Any] = {}
        self._executor    = None
        self._shap_explainer = None
        self._drift_detectors: Dict[ModelName, Any] = {}

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def load(self) -> None:
        import concurrent.futures
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(self._executor, self._load_distilbert),
            loop.run_in_executor(self._executor, self._load_bilstm),
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Init drift detectors
        from ml.evaluation.drift_detector import DriftDetector
        for model_name in self._models:
            self._drift_detectors[model_name] = DriftDetector(model_name.value)

        log.info("model_service.ready", loaded=self.loaded_models)

    async def unload(self) -> None:
        if self._executor:
            self._executor.shutdown(wait=False)

    async def reload(self, model_name: ModelName) -> None:
        loop = asyncio.get_running_loop()
        loader = self._load_distilbert if model_name == ModelName.DISTILBERT else self._load_bilstm
        await loop.run_in_executor(self._executor, loader)
        log.info("model.reloaded", model=model_name)

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
        model_name = self._resolve_model(model_name)
        loop = asyncio.get_running_loop()
        fn   = functools.partial(self._sync_predict, text, model_name, explain)
        result = await loop.run_in_executor(self._executor, fn)

        # Record to drift detector (non-blocking)
        detector = self._drift_detectors.get(model_name)
        if detector:
            detector.record(result["sentiment"].value, result["confidence"])

        # Prometheus metrics
        PREDICTION_COUNTER.labels(model=model_name.value, sentiment=result["sentiment"].value).inc()
        CONFIDENCE_HISTOGRAM.labels(model=model_name.value).observe(result["confidence"])

        return result

    async def batch_predict(
        self,
        texts: List[str],
        model_name: ModelName,
    ) -> List[Dict[str, Any]]:
        model_name = self._resolve_model(model_name)
        loop = asyncio.get_running_loop()
        fn   = functools.partial(self._sync_batch_predict, texts, model_name)
        results = await loop.run_in_executor(self._executor, fn)

        # Batch drift recording
        detector = self._drift_detectors.get(model_name)
        if detector:
            detector.record_batch(
                [r["sentiment"].value for r in results],
                [r["confidence"]      for r in results],
            )
        return results

    def get_drift_stats(self, model_name: ModelName) -> dict:
        detector = self._drift_detectors.get(model_name)
        return detector.stats if detector else {}

    async def check_drift(self, model_name: ModelName, force: bool = False):
        detector = self._drift_detectors.get(model_name)
        if detector:
            return detector.check_drift(force=force)
        return None

    # ── Sync inference ─────────────────────────────────────────────────────

    def _sync_predict(self, text: str, model_name: ModelName, explain: bool) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if model_name == ModelName.DISTILBERT:
            result = self._distilbert_predict([text], explain)[0]
        else:
            result = self._bilstm_predict([text])[0]
        PREDICTION_LATENCY.labels(model=model_name.value).observe(time.perf_counter() - t0)
        return result

    def _sync_batch_predict(self, texts: List[str], model_name: ModelName) -> List[Dict[str, Any]]:
        if model_name == ModelName.DISTILBERT:
            return self._distilbert_predict(texts, explain=False)
        return self._bilstm_predict(texts)

    # ── DistilBERT inference ───────────────────────────────────────────────

    def _distilbert_predict(self, texts: List[str], explain: bool) -> List[Dict[str, Any]]:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from ml.preprocessing.text_cleaner import clean_for_distilbert

        tokenizer = self._tokenizers[ModelName.DISTILBERT]
        model     = self._models[ModelName.DISTILBERT]

        cleaned = [clean_for_distilbert(t) for t in texts]
        inputs  = tokenizer(
            cleaned, return_tensors="pt",
            truncation=True, padding=True,
            max_length=settings.MAX_SEQUENCE_LENGTH,
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs_np = F.softmax(logits, dim=-1).numpy()

        results = []
        for i, (prob, orig_text) in enumerate(zip(probs_np, texts)):
            label_idx  = int(np.argmax(prob))
            sentiment  = LABEL_MAP.get(label_idx, SentimentLabel.NEUTRAL)
            result = {
                "text":         orig_text,
                "sentiment":    sentiment,
                "confidence":   round(float(prob[label_idx]), 4),
                "probabilities": {
                    SentimentLabel.NEGATIVE:   round(float(prob[0]), 4),
                    SentimentLabel.POSITIVE:   round(float(prob[1]), 4),
                    SentimentLabel.NEUTRAL:    round(float(prob[2]), 4),
                    SentimentLabel.IRRELEVANT: round(float(prob[3]), 4),
                },
            }
            if explain and i == 0:
                result["explanation"] = self._explain_distilbert(orig_text)
            results.append(result)
        return results

    # ── Bi-LSTM inference ─────────────────────────────────────────────────

    def _bilstm_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        import numpy as np
        import tensorflow as tf
        from ml.preprocessing.text_cleaner import clean_for_bilstm

        tokenizer = self._tokenizers[ModelName.BILSTM]
        model     = self._models[ModelName.BILSTM]

        cleaned   = [clean_for_bilstm(t) for t in texts]
        sequences = tokenizer.texts_to_sequences(cleaned)
        padded    = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=settings.MAX_SEQUENCE_LENGTH, padding="post", truncating="post"
        )
        probs_batch = model.predict(padded, verbose=0)

        results = []
        all_labels = [SentimentLabel.NEGATIVE, SentimentLabel.POSITIVE,
                      SentimentLabel.NEUTRAL, SentimentLabel.IRRELEVANT]
        for text, prob in zip(texts, probs_batch):
            label_idx = int(np.argmax(prob))
            results.append({
                "text":         text,
                "sentiment":    LABEL_MAP.get(label_idx, SentimentLabel.NEUTRAL),
                "confidence":   round(float(prob[label_idx]), 4),
                "probabilities": {label: round(float(prob[j]), 4) for j, label in enumerate(all_labels)},
            })
        return results

    # ── SHAP explanation ───────────────────────────────────────────────────

    def _explain_distilbert(self, text: str) -> List[Dict[str, Any]]:
        try:
            from ml.explainability.shap_explainer import SHAPExplainer
            if self._shap_explainer is None:
                self._shap_explainer = SHAPExplainer(
                    tokenizers={m.value: t for m, t in self._tokenizers.items()},
                    models={m.value: mdl for m, mdl in self._models.items()},
                )
            result = self._shap_explainer.explain_distilbert(text)
            return self._shap_explainer.to_api_format(result)
        except Exception as e:
            log.warning("explain.failed", error=str(e))
            return []

    # ── Loaders ───────────────────────────────────────────────────────────

    def _resolve_model(self, model_name: ModelName) -> ModelName:
        if model_name in self._models:
            return model_name
        fallback = ModelName.BILSTM if model_name == ModelName.DISTILBERT else ModelName.DISTILBERT
        if fallback in self._models:
            log.warning("model.fallback", requested=model_name.value, using=fallback.value)
            return fallback
        raise RuntimeError(f"No models loaded. Requested: {model_name}")

    def _load_distilbert(self) -> None:
        model_path = Path(settings.DISTILBERT_MODEL_PATH)
        if not model_path.exists():
            log.warning("distilbert.not_found", path=str(model_path))
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            log.info("distilbert.loading")
            self._tokenizers[ModelName.DISTILBERT] = AutoTokenizer.from_pretrained(str(model_path))
            self._models[ModelName.DISTILBERT]     = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            self._models[ModelName.DISTILBERT].eval()
            log.info("distilbert.loaded")
        except Exception as e:
            log.error("distilbert.load_error", error=str(e))
            raise

    def _load_bilstm(self) -> None:
        model_path = Path(settings.BILSTM_MODEL_PATH)
        tok_path   = Path(settings.BILSTM_TOKENIZER_PATH)
        if not model_path.exists():
            log.warning("bilstm.not_found", path=str(model_path))
            return
        try:
            import tensorflow as tf
            log.info("bilstm.loading")
            self._models[ModelName.BILSTM]     = tf.keras.models.load_model(str(model_path))
            with open(tok_path, "rb") as f:
                self._tokenizers[ModelName.BILSTM] = pickle.load(f)
            log.info("bilstm.loaded")
        except Exception as e:
            log.error("bilstm.load_error", error=str(e))
            raise
