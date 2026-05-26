"""
Text preprocessing pipeline.
Handles Twitter-specific noise: URLs, mentions, hashtags, emojis, slang.
Two pipelines:
  - clean_for_bilstm(): custom tokenizer-compatible cleaning
  - clean_for_distilbert(): lighter cleaning (DistilBERT handles subword tokenisation)
"""

import re
import unicodedata
from typing import List


# ── Regex patterns (compiled once for performance) ────────────────────────
_URL_RE        = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE    = re.compile(r"@\w+")
_HASHTAG_RE    = re.compile(r"#(\w+)")
_HTML_RE       = re.compile(r"<[^>]+>")
_REPEATED_RE   = re.compile(r"(.)\1{2,}")   # "sooooo" → "soo"
_WHITESPACE_RE = re.compile(r"\s+")
_NON_ASCII_RE  = re.compile(r"[^\x00-\x7F]+")
_EMOJI_RE      = re.compile(
    "[\U00010000-\U0010ffff"
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U00002702-\U000027B0]+",
    flags=re.UNICODE,
)

# Common Twitter abbreviations
SLANG_MAP = {
    "u": "you", "r": "are", "ur": "your", "n": "and",
    "omg": "oh my god", "lol": "laughing out loud", "wtf": "what the",
    "imo": "in my opinion", "imho": "in my humble opinion",
    "tbh": "to be honest", "afaik": "as far as i know",
    "smh": "shaking my head", "fwiw": "for what it is worth",
}


def clean_for_bilstm(text: str) -> str:
    """
    Aggressive cleaning for Bi-LSTM with custom vocabulary.
    Normalises to lowercase ASCII tokens the trained tokeniser understands.
    """
    text = _HTML_RE.sub("", text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(r" \1 ", text)  # keep hashtag text
    text = _EMOJI_RE.sub(" ", text)
    text = unicodedata.normalize("NFKD", text)
    text = _NON_ASCII_RE.sub("", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = _REPEATED_RE.sub(r"\1\1", text)   # reduce elongation but keep some emphasis
    # Expand slang
    words = text.split()
    words = [SLANG_MAP.get(w, w) for w in words]
    text = " ".join(words)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_for_distilbert(text: str) -> str:
    """
    Lighter cleaning for DistilBERT — preserve more context since
    the transformer's subword tokeniser handles morphological variation.
    We still remove noise that carries no semantic signal.
    """
    text = _HTML_RE.sub("", text)
    text = _URL_RE.sub("[URL]", text)       # replace, don't delete — DistilBERT sees [URL]
    text = _MENTION_RE.sub("[USER]", text)  # preserve mention count as signal
    text = _HASHTAG_RE.sub(r"\1", text)     # strip # but keep text
    text = _EMOJI_RE.sub(" ", text)
    text = _REPEATED_RE.sub(r"\1\1\1", text)  # allow up to 3 repeats
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def batch_clean(texts: List[str], model: str = "distilbert") -> List[str]:
    """Clean a list of texts in bulk."""
    fn = clean_for_distilbert if model == "distilbert" else clean_for_bilstm
    return [fn(t) for t in texts]


# ── Unit-testable preprocessing audit ────────────────────────────────────

def audit_preprocessing(text: str) -> dict:
    """
    Returns a diff showing what each cleaning step removed/changed.
    Useful for debugging and for the explainability dashboard.
    """
    return {
        "original": text,
        "bilstm_cleaned": clean_for_bilstm(text),
        "distilbert_cleaned": clean_for_distilbert(text),
        "urls_found": len(_URL_RE.findall(text)),
        "mentions_found": len(_MENTION_RE.findall(text)),
        "hashtags_found": _HASHTAG_RE.findall(text),
        "emojis_found": _EMOJI_RE.findall(text),
    }


if __name__ == "__main__":
    # Quick smoke test
    sample = "@Apple just released #iPhone16 🔥🔥 sooooo good!!! Check https://apple.com"
    print("Original:   ", sample)
    print("BiLSTM:     ", clean_for_bilstm(sample))
    print("DistilBERT: ", clean_for_distilbert(sample))
    print("Audit:      ", audit_preprocessing(sample))
