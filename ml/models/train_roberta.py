"""
Fine-tune RoBERTa-base for Twitter sentiment classification.

Why RoBERTa over BERT/DistilBERT?
  - Trained longer with more data and no NSP task
  - Uses BPE tokenisation (better for Twitter slang/emoji)
  - 88.5M params vs 66M DistilBERT — accuracy gain worth it for batch workloads
  - twitter-roberta-base-sentiment is already pre-trained on tweets!

Model: cardiffnlp/twitter-roberta-base-sentiment-latest
  → Directly fine-tuned on 124M tweets — ideal starting point.

Usage:
    python -m ml.models.train_roberta \
        --data_path twitter_training.csv \
        --output_dir ml/saved_models/roberta \
        --epochs 5
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ml.preprocessing.text_cleaner import clean_for_distilbert  # same cleaning works for RoBERTa

TWITTER_ROBERTA_CHECKPOINT = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def load_data(data_path: str):
    df = pd.read_csv(data_path, header=None, names=["id", "entity", "sentiment", "text"], dtype=str)
    df = df.dropna(subset=["text", "sentiment"])
    df["text"] = df["text"].apply(clean_for_distilbert)
    df = df[df["text"].str.strip().str.len() > 3]
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["sentiment"])
    return df["text"].tolist(), df["label"].tolist(), le


def compute_metrics(eval_pred):
    import numpy as np
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":    accuracy_score(labels, preds),
        "f1_macro":    f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def train(args):
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
    import torch
    from torch.utils.data import Dataset

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("sentiment-analysis")

    class TweetDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.enc = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self):      return len(self.labels)
        def __getitem__(self, i): return {k: v[i] for k, v in self.enc.items()} | {"labels": self.labels[i]}

    with mlflow.start_run(run_name=f"roberta-twitter-ep{args.epochs}-{int(time.time())}") as run:
        print(f"MLflow run: {run.info.run_id}")

        texts, labels, le = load_data(args.data_path)
        num_classes = len(le.classes_)

        X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.15, stratify=labels, random_state=42)
        X_tr, X_vl, y_tr, y_vl = train_test_split(X_tr,  y_tr,  test_size=0.15, stratify=y_tr,  random_state=42)

        mlflow.log_params({
            "model_checkpoint": TWITTER_ROBERTA_CHECKPOINT,
            "num_labels": num_classes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_length": args.max_len,
            "train_size": len(X_tr),
        })

        tokenizer = AutoTokenizer.from_pretrained(TWITTER_ROBERTA_CHECKPOINT)
        model = AutoModelForSequenceClassification.from_pretrained(
            TWITTER_ROBERTA_CHECKPOINT,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,  # classifier head is replaced
        )

        tr_ds = TweetDataset(X_tr, y_tr, tokenizer, args.max_len)
        vl_ds = TweetDataset(X_vl, y_vl, tokenizer, args.max_len)
        te_ds = TweetDataset(X_te, y_te, tokenizer, args.max_len)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            learning_rate=args.lr,
            warmup_ratio=0.06,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_weighted",
            report_to="none",
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=2,
            label_smoothing_factor=0.1,  # reduces overconfidence
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tr_ds,
            eval_dataset=vl_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        trainer.train()

        # Test evaluation
        out   = trainer.predict(te_ds)
        preds = np.argmax(out.predictions, axis=-1)
        acc   = accuracy_score(y_te, preds)
        f1w   = f1_score(y_te, preds, average="weighted")
        report = classification_report(y_te, preds, target_names=le.classes_)

        print(f"\nTest Accuracy: {acc:.4f} | F1 (weighted): {f1w:.4f}")
        print(report)

        mlflow.log_metrics({"test_accuracy": round(acc, 4), "test_f1_weighted": round(f1w, 4)})
        mlflow.log_text(report, "test_classification_report.txt")

        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        mlflow.log_artifacts(str(output_dir), artifact_path="roberta_model")

        print(f"Model saved: {output_dir} | MLflow: {run.info.run_id}")
        return run.info.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  default="twitter_training.csv")
    parser.add_argument("--output_dir", default="ml/saved_models/roberta")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-5)
    parser.add_argument("--max_len",    type=int,   default=128)
    args = parser.parse_args()
    train(args)
