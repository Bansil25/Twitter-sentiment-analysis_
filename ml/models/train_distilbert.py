"""
Fine-tune DistilBERT for Twitter sentiment classification.
Tracks every experiment with MLflow — metrics, params, artifacts.

Usage:
    python -m ml.models.train_distilbert \
        --data_path twitter_training.csv \
        --output_dir ml/saved_models/distilbert \
        --epochs 3 \
        --batch_size 32
"""

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from ml.preprocessing.text_cleaner import clean_for_distilbert

LABEL_NAMES = ["Negative", "Positive", "Neutral", "Irrelevant"]
MODEL_CHECKPOINT = "distilbert-base-uncased"


# ── Dataset ────────────────────────────────────────────────────────────────

class TwitterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_length, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()} | {"labels": self.labels[idx]}


# ── Training ───────────────────────────────────────────────────────────────

def load_and_prepare_data(data_path: str):
    """Load twitter_training.csv, clean text, encode labels."""
    df = pd.read_csv(data_path, header=None, names=["id", "entity", "sentiment", "text"])
    df = df.dropna(subset=["text", "sentiment"])
    df["text"] = df["text"].astype(str).apply(clean_for_distilbert)
    df = df[df["text"].str.len() > 5]

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["sentiment"])
    print(f"Loaded {len(df)} samples. Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return df["text"].tolist(), df["label"].tolist(), le


def compute_metrics(eval_pred):
    """Called by HuggingFace Trainer after each eval epoch."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def train(args):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("sentiment-analysis")

    with mlflow.start_run(run_name=f"distilbert-ft-ep{args.epochs}") as run:
        print(f"MLflow run ID: {run.info.run_id}")

        # Log hyperparameters
        mlflow.log_params({
            "model_checkpoint": MODEL_CHECKPOINT,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_length": args.max_length,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
        })

        # Data
        texts, labels, label_encoder = load_and_prepare_data(args.data_path)
        # Save the label encoder right after data prep — single source of truth
        import pickle as _pickle
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.output_dir) / "label_encoder.pkl", "wb") as f:
            _pickle.dump(label_encoder, f)
        print(f"Saved label_encoder.pkl with classes: {list(label_encoder.classes_)}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )
        mlflow.log_params({"train_size": len(X_train), "val_size": len(X_val), "test_size": len(X_test)})

        # Tokenizer + model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_CHECKPOINT,
            num_labels=len(set(labels)),
        )

        train_ds = TwitterDataset(X_train, y_train, tokenizer, args.max_length)
        val_ds   = TwitterDataset(X_val,   y_val,   tokenizer, args.max_length)
        test_ds  = TwitterDataset(X_test,  y_test,  tokenizer, args.max_length)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            learning_rate=args.lr,
            warmup_ratio=0.1,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            report_to="none",  # We log to MLflow manually
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=2,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        # Train
        trainer.train()

        # Evaluate on test set
        preds_output = trainer.predict(test_ds)
        preds = np.argmax(preds_output.predictions, axis=-1)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        report = classification_report(y_test, preds, target_names=label_encoder.classes_)
        cm = confusion_matrix(y_test, preds)

        print(f"\n{'='*60}")
        print(f"Test Accuracy: {acc:.4f}  |  F1 (weighted): {f1:.4f}")
        print(f"\n{report}")
        print(f"Confusion Matrix:\n{cm}")

        mlflow.log_metrics({"test_accuracy": acc, "test_f1_weighted": f1})
        mlflow.log_text(report, "classification_report.txt")

        # Save confusion matrix as artifact
        _save_confusion_matrix(cm, label_encoder.classes_, args.output_dir)
        mlflow.log_artifact(f"{args.output_dir}/confusion_matrix.png")

        # Save model to HuggingFace format (also logged to MLflow)
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        mlflow.log_artifacts(str(output_path), artifact_path="distilbert_model")

        print(f"\nModel saved to: {output_path}")
        print(f"MLflow run: {run.info.run_id}")
        return run.info.run_id


def _save_confusion_matrix(cm, class_names, output_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — DistilBERT")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  default="twitter_training.csv")
    parser.add_argument("--output_dir", default="ml/saved_models/distilbert")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--max_length", type=int,   default=128)
    args = parser.parse_args()
    train(args)
