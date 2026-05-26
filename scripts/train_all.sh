#!/usr/bin/env bash
# ============================================================
# train_all.sh — Train all models in the right order.
# Run from the project root:
#   chmod +x scripts/train_all.sh && ./scripts/train_all.sh
# ============================================================

set -e  # Exit on error

DATA_PATH="${DATA_PATH:-twitter_training.csv}"
MLFLOW_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"

echo "=================================================="
echo " SentimentAI — Full Training Pipeline"
echo "=================================================="
echo " Data:       $DATA_PATH"
echo " MLflow:     $MLFLOW_URI"
echo ""

# ── Step 1: Bi-LSTM (fast baseline, ~15 min on CPU) ──────────────────────
echo "[1/4] Training Bi-LSTM baseline..."
python -m ml.models.train_bilstm \
    --data_path "$DATA_PATH" \
    --output_dir ml/saved_models/bilstm \
    --epochs 15 \
    --batch_size 128 \
    --lr 1e-3
echo "✓ Bi-LSTM complete"

# ── Step 2: DistilBERT fine-tuning (~40 min on CPU, ~8 min on GPU) ───────
echo ""
echo "[2/4] Fine-tuning DistilBERT..."
python -m ml.models.train_distilbert \
    --data_path "$DATA_PATH" \
    --output_dir ml/saved_models/distilbert \
    --epochs 3 \
    --batch_size 32 \
    --lr 2e-5
echo "✓ DistilBERT complete"

# ── Step 3: RoBERTa fine-tuning (optional, ~60 min on CPU) ───────────────
if [ "${SKIP_ROBERTA:-false}" != "true" ]; then
    echo ""
    echo "[3/4] Fine-tuning RoBERTa (set SKIP_ROBERTA=true to skip)..."
    python -m ml.models.train_roberta \
        --data_path "$DATA_PATH" \
        --output_dir ml/saved_models/roberta \
        --epochs 5 \
        --batch_size 16 \
        --lr 1e-5
    echo "✓ RoBERTa complete"
else
    echo "[3/4] Skipping RoBERTa (SKIP_ROBERTA=true)"
fi

# ── Step 4: Full evaluation and comparison ───────────────────────────────
echo ""
echo "[4/4] Running evaluation suite..."
python -m ml.evaluation.evaluate_all \
    --data_path "$DATA_PATH" \
    --output_dir ml/evaluation/reports
echo "✓ Evaluation complete"

echo ""
echo "=================================================="
echo " Training complete!"
echo " → Model artifacts:  ml/saved_models/"
echo " → Evaluation plots: ml/evaluation/reports/"
echo " → MLflow UI:        $MLFLOW_URI"
echo "=================================================="
