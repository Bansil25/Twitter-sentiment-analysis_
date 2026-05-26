"""
Explainability module using SHAP.

Two levels:
  1. Local explanations  — per-prediction token importance (the API's explain=true)
  2. Global explanations — dataset-level feature importance (for the dashboard)

Both models supported:
  - DistilBERT: SHAP PartitionExplainer (text masking approach)
  - Bi-LSTM:    SHAP DeepExplainer (gradient-based, faster)

Usage:
    from ml.explainability.shap_explainer import SHAPExplainer
    explainer = SHAPExplainer(model_service)
    result = explainer.explain_distilbert("Apple's new phone is amazing!")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class TokenExplanation:
    """Single token with its SHAP attribution score."""
    token: str
    score: float          # positive = pushes toward predicted class
    abs_score: float = field(init=False)
    polarity: str = field(init=False)

    def __post_init__(self):
        self.abs_score = abs(self.score)
        self.polarity  = "positive" if self.score > 0 else "negative"


@dataclass
class ExplanationResult:
    text: str
    predicted_class: str
    confidence: float
    tokens: List[TokenExplanation]
    top_positive_tokens: List[str]  # tokens most responsible for predicted class
    top_negative_tokens: List[str]  # tokens pushing against predicted class
    model_name: str


class SHAPExplainer:
    """
    Wraps SHAP explainers for both DistilBERT and Bi-LSTM.
    Instantiated once and reused across requests.
    """

    def __init__(self, tokenizers: dict, models: dict):
        self.tokenizers = tokenizers
        self.models     = models
        self._distilbert_explainer = None  # lazy-init (SHAP setup takes ~1s)
        self._bilstm_explainer     = None

    # ── DistilBERT explanation ─────────────────────────────────────────────

    def explain_distilbert(
        self,
        text: str,
        target_class_idx: Optional[int] = None,
        max_tokens: int = 30,
    ) -> ExplanationResult:
        """
        Use SHAP PartitionExplainer for DistilBERT.
        This masks token groups and measures output change — model-agnostic,
        but slower (~300ms). Cache the explainer object across calls.
        """
        import shap
        import torch
        import torch.nn.functional as F

        tokenizer = self.tokenizers.get("distilbert")
        model     = self.models.get("distilbert")
        if not tokenizer or not model:
            raise RuntimeError("DistilBERT model not loaded")

        def predict_fn(texts: list) -> np.ndarray:
            """Wrapper that SHAP calls with masked text variants."""
            inputs = tokenizer(
                list(texts), return_tensors="pt",
                truncation=True, padding=True, max_length=128,
            )
            with torch.no_grad():
                logits = model(**inputs).logits
            return F.softmax(logits, dim=-1).numpy()

        # Build explainer (lazy init — expensive)
        if self._distilbert_explainer is None:
            self._distilbert_explainer = shap.Explainer(predict_fn, tokenizer)

        shap_values = self._distilbert_explainer([text])
        tokens = [t.replace("Ġ", " ").strip() for t in shap_values.data[0]]

        # Get prediction first
        base_probs = predict_fn([text])[0]
        pred_idx   = int(np.argmax(base_probs)) if target_class_idx is None else target_class_idx
        scores     = shap_values.values[0, :, pred_idx]

        # Build token explanations (trim to max_tokens, skip padding)
        token_explanations = []
        for token, score in zip(tokens[:max_tokens], scores[:max_tokens]):
            if token in {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}:
                continue
            if not token.strip():
                continue
            token_explanations.append(TokenExplanation(token=token, score=round(float(score), 4)))

        # Sort by absolute impact for top lists
        sorted_by_abs  = sorted(token_explanations, key=lambda t: t.abs_score, reverse=True)
        top_positive   = [t.token for t in sorted_by_abs if t.score > 0][:5]
        top_negative   = [t.token for t in sorted_by_abs if t.score < 0][:5]

        label_map = {0: "Negative", 1: "Positive", 2: "Neutral", 3: "Irrelevant"}
        return ExplanationResult(
            text=text,
            predicted_class=label_map.get(pred_idx, str(pred_idx)),
            confidence=round(float(base_probs[pred_idx]), 4),
            tokens=token_explanations,
            top_positive_tokens=top_positive,
            top_negative_tokens=top_negative,
            model_name="distilbert",
        )

    # ── Bi-LSTM explanation ────────────────────────────────────────────────

    def explain_bilstm(
        self,
        text: str,
        background_texts: Optional[List[str]] = None,
        max_tokens: int = 30,
    ) -> ExplanationResult:
        """
        Use SHAP DeepExplainer for Bi-LSTM (gradient-based).
        Requires a background dataset to establish baseline.
        Faster than PartitionExplainer (~50ms vs ~300ms).
        """
        import shap
        import tensorflow as tf
        from ml.preprocessing.text_cleaner import clean_for_bilstm

        tokenizer = self.tokenizers.get("bilstm")
        model     = self.models.get("bilstm")
        if not tokenizer or not model:
            raise RuntimeError("Bi-LSTM model not loaded")

        # Preprocess text
        cleaned  = clean_for_bilstm(text)
        seq      = tokenizer.texts_to_sequences([cleaned])
        padded   = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=128, padding="post")

        # Background sample (needed for DeepExplainer baseline)
        if background_texts is None:
            background = np.zeros((50, 128), dtype="int32")  # zero-pad baseline
        else:
            bg_cleaned  = [clean_for_bilstm(t) for t in background_texts[:50]]
            bg_seqs     = tokenizer.texts_to_sequences(bg_cleaned)
            background  = tf.keras.preprocessing.sequence.pad_sequences(bg_seqs, maxlen=128, padding="post")

        if self._bilstm_explainer is None:
            self._bilstm_explainer = shap.DeepExplainer(model, background)

        shap_values = self._bilstm_explainer.shap_values(padded)
        probs       = model.predict(padded, verbose=0)[0]
        pred_idx    = int(np.argmax(probs))

        # SHAP values shape: (n_classes, batch, seq_len, embed_dim)
        # Reduce over embedding dimension to get per-token importance
        token_scores = np.abs(shap_values[pred_idx][0]).sum(axis=-1)  # (seq_len,)
        # Normalise to signed via original contribution
        signed_scores = shap_values[pred_idx][0].sum(axis=-1)

        # Map back to original tokens
        token_ids = padded[0]
        inv_index = {v: k for k, v in tokenizer.word_index.items()}
        word_tokens = [inv_index.get(int(tid), "[UNK]") for tid in token_ids if int(tid) != 0]

        token_explanations = []
        for word, score in zip(word_tokens[:max_tokens], signed_scores[:len(word_tokens)][:max_tokens]):
            if word in {"[PAD]", "[OOV]", "<OOV>"}:
                continue
            token_explanations.append(TokenExplanation(token=word, score=round(float(score), 4)))

        sorted_by_abs = sorted(token_explanations, key=lambda t: t.abs_score, reverse=True)
        top_positive  = [t.token for t in sorted_by_abs if t.score > 0][:5]
        top_negative  = [t.token for t in sorted_by_abs if t.score < 0][:5]

        label_map = {0: "Negative", 1: "Positive", 2: "Neutral", 3: "Irrelevant"}
        return ExplanationResult(
            text=text,
            predicted_class=label_map.get(pred_idx, str(pred_idx)),
            confidence=round(float(probs[pred_idx]), 4),
            tokens=token_explanations,
            top_positive_tokens=top_positive,
            top_negative_tokens=top_negative,
            model_name="bilstm",
        )

    # ── Serialisation for the API ──────────────────────────────────────────

    def to_api_format(self, result: ExplanationResult) -> list:
        """Convert to the format expected by app/schemas/sentiment.py."""
        return [{"token": t.token, "score": t.score} for t in result.tokens]


# ── Standalone visualisation (for notebooks + dashboard screenshots) ───────

def visualise_explanation(result: ExplanationResult, output_path: Optional[str] = None):
    """
    Generate an HTML highlight visualisation.
    Positive-impact tokens → green background, negative → red.
    """
    if not result.tokens:
        return "<p>No explanation available.</p>"

    max_abs = max(abs(t.score) for t in result.tokens) or 1.0
    parts   = []
    for t in result.tokens:
        intensity = abs(t.score) / max_abs
        if t.score > 0:
            r, g, b = int(255 * (1 - intensity * 0.6)), 255, int(255 * (1 - intensity * 0.6))
        else:
            r, g, b = 255, int(255 * (1 - intensity * 0.6)), int(255 * (1 - intensity * 0.6))
        style = f"background-color: rgba({r},{g},{b},0.8); padding: 2px 3px; border-radius: 3px; margin: 1px;"
        title = f"SHAP score: {t.score:+.4f}"
        parts.append(f'<span style="{style}" title="{title}">{t.token}</span>')

    html = f"""
    <div style="font-family: sans-serif; font-size: 14px; line-height: 2;">
        <p><strong>Predicted:</strong> {result.predicted_class}
           &nbsp; <strong>Confidence:</strong> {result.confidence:.1%}</p>
        <p style="line-height: 2.2;">{" ".join(parts)}</p>
        <p style="font-size: 12px; color: #666;">
            🟢 Green = pushes toward {result.predicted_class} &nbsp;|&nbsp;
            🔴 Red = pushes against {result.predicted_class}
        </p>
    </div>
    """
    if output_path:
        with open(output_path, "w") as f:
            f.write(html)
    return html
