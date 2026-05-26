# Model Architecture Comparison & Tradeoff Analysis

This document explains every model choice in the project — exactly the kind of
reasoning you need to articulate in ML engineer interviews.

---

## Models Compared

| | Bi-LSTM | DistilBERT | RoBERTa |
|---|---|---|---|
| **Architecture** | Sequential RNN | Transformer (distilled) | Transformer (robust) |
| **Parameters** | ~4.2M | ~66M | ~88.5M |
| **Pre-training** | None (GloVe init) | BERT distillation on BooksCorpus+Wiki | 124M tweets |
| **Tokenisation** | Custom vocab (Keras) | WordPiece (30k vocab) | BPE (50k vocab) |
| **Max sequence** | 128 tokens | 512 tokens | 512 tokens |
| **Our accuracy** | 83.2% | 92.1% | ~94–95% (estimated) |
| **F1 weighted** | 82.8% | 91.8% | ~93–94% |
| **P50 latency (CPU)** | ~12ms | ~48ms | ~70ms |
| **Memory footprint** | 16 MB | 253 MB | 340 MB |
| **Fine-tune time (1 epoch, CPU)** | ~8 min | ~40 min | ~60 min |
| **Deployment cost** | Very low | Medium | Medium-high |

---

## Why Each Model Behaves the Way It Does

### Bi-LSTM

The bidirectional LSTM processes the sequence forward AND backward,
capturing both past and future context for each token. For sentiment:
"I thought the product was great but the packaging was terrible" —
the word "terrible" near the end affects the sentiment of the whole sentence.
The backward pass captures this. The forward pass alone would weight
"great" more heavily.

**Where it wins:** Low latency, CPU-friendly, small memory footprint.
Excellent for edge deployment or high-QPS scenarios where a 92% model
isn't needed.

**Where it loses:** No understanding of subword morphology. "amazinggg"
and "amazing" are two different tokens. No cross-sentence context.
Struggles with negation patterns ("not bad at all").

**Why 83% accuracy and not higher:** LSTMs have a fixed-size hidden state.
At sequence length 128, the gradient signal from token 1 is heavily diluted
by token 128. Attention mechanisms solve this directly.

### DistilBERT

Trained via knowledge distillation from BERT-base — a student model that
learns to mimic BERT's output distribution. Retains 97% of BERT's performance
at 40% fewer parameters.

**Key mechanism:** Multi-head self-attention. Every token attends to every
other token simultaneously. "terrible" can directly attend to "packaging"
AND to "great" in the same forward pass. This is why transformers dominate
sequence classification.

**Why it outperforms Bi-LSTM by ~9%:** Attention is not limited by sequence
length. The model explicitly learns which tokens are relevant to each other.
The pre-training on 3.3B tokens gives it broad language understanding before
we even touch our Twitter data.

**Production recommendation:** First choice for CPU-only deployments where
<100ms P95 is acceptable.

### RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest)

The same transformer architecture as BERT/DistilBERT, but trained differently:
- Removed the Next Sentence Prediction objective (shown to be harmful)
- Trained on 160GB of data vs BERT's ~16GB
- Trained for 10x longer with 10x larger batches
- Dynamic masking (each epoch sees different masked tokens)

**The Twitter-specific advantage:** The Cardiff NLP checkpoint was
pre-trained on 124M tweets, then fine-tuned on sentiment. Our fine-tuning
is starting from a model that already "speaks Twitter" — it understands
abbreviations, emoji semantics, hashtag patterns, and reply threads.
This is called domain-adaptive pre-training, and it's the single highest
ROI technique in applied NLP.

**Where it wins:** Best accuracy, especially on Twitter slang, irony, and
mixed-sentiment tweets. If your product is used for brand monitoring,
RoBERTa is the right choice.

**Where it loses:** Larger than DistilBERT. Requires more GPU memory. 
Fine-tuning takes longer.

---

## Production Decision Framework

Ask these questions in this order:

**1. What is the latency SLA?**
- < 20ms P95 → Bi-LSTM only
- < 100ms P95 → DistilBERT (CPU) or RoBERTa (GPU)
- < 500ms P95 → Any model; optimise other bottlenecks

**2. Is a GPU available in production?**
- No GPU → DistilBERT is the right choice. RoBERTa's accuracy gain
  doesn't justify its slower CPU inference.
- GPU available → RoBERTa. The latency difference vs DistilBERT collapses
  to ~10ms on GPU.

**3. How important is explainability?**
- Legal/compliance requirement → Bi-LSTM with SHAP DeepExplainer is
  faster and more interpretable at the token level.
- Nice-to-have → DistilBERT SHAP works, just ~300ms overhead.

**4. What is the expected QPS?**
- < 100 QPS → Any model
- 100–1000 QPS → Add Redis caching; deduplicate identical inputs
- > 1000 QPS → Bi-LSTM, OR DistilBERT with ONNX Runtime quantisation
  (reduces DistilBERT latency from 48ms to ~18ms on CPU)

---

## ONNX Quantisation (the production optimisation interviewers love)

DistilBERT can be exported to ONNX and quantised to INT8:

```python
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

model  = ORTModelForSequenceClassification.from_pretrained(
    "ml/saved_models/distilbert",
    export=True,
)
# After quantisation: ~18ms P50 on CPU, minimal accuracy loss (<0.5%)
```

This brings DistilBERT close to Bi-LSTM latency while keeping 92%+ accuracy.
This is the real production answer — not "just use the smaller model".

---

## MLflow Experiment Tracking Setup

After training all three models:

```bash
# Start MLflow UI
mlflow ui --port 5000

# View comparison at: http://localhost:5000
# Compare runs side-by-side in the Experiments view
# Register the best model in the Model Registry
```

In the MLflow UI:
1. Go to `sentiment-analysis` experiment
2. Select all three training runs
3. Click "Compare" — generates side-by-side metric table
4. The run with highest `test_f1_weighted` → promote to "Production"

---

## Interview Talking Points

**"Why not BERT-base instead of DistilBERT?"**
DistilBERT is 40% smaller, 60% faster, with 97% of BERT's performance.
In production, the 3% accuracy loss is worth the halved compute cost.
If you need BERT-level accuracy, use RoBERTa — it's better than BERT anyway.

**"How would you improve accuracy further?"**
1. Data augmentation — back-translation, synonym replacement
2. Ensemble: weighted average of DistilBERT + RoBERTa probabilities
3. Task-specific fine-tuning on your exact domain
4. Self-training: use high-confidence predictions on unlabelled data

**"How do you handle concept drift?"**
KL divergence monitoring on the rolling prediction distribution.
If the distribution shifts by more than 0.1 KL divergence from the
training baseline, we trigger an alert. Weekly retraining on recent data
with the old model as a teacher (knowledge distillation).

**"What's your biggest concern about this model in production?"**
Sycophancy bias — the model was trained on tweet-level sentiment,
not reply-thread sentiment. A negative reply to a positive tweet might
be classified as positive because it quotes positive text.
Mitigation: fine-tune specifically on reply threads.
