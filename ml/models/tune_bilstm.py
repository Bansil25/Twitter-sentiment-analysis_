"""
Hyperparameter optimisation for Bi-LSTM using Optuna.
Runs N trials, logs every trial to MLflow, returns best config.

Why Optuna over Grid Search?
  - Bayesian optimisation (TPE sampler) — much more efficient than grid search
  - Pruning: bad trials are stopped early (saves hours of compute)
  - Native MLflow integration
  - Parallel trials support

Usage:
    python -m ml.models.tune_bilstm \
        --data_path twitter_training.csv \
        --n_trials 30 \
        --timeout_minutes 120
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import numpy as np
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ml.preprocessing.text_cleaner import clean_for_bilstm


def objective(trial, X_train_pad, y_train, X_val_pad, y_val, vocab_size, num_classes, run_prefix: str):
    """Optuna objective function — returns validation F1 (to maximise)."""
    import tensorflow as tf

    # ── Hyperparameter search space ───────────────────────────────────────
    lstm_units_1 = trial.suggest_categorical("lstm_units_1", [64, 128, 256])
    lstm_units_2 = trial.suggest_categorical("lstm_units_2", [32, 64, 128])
    spatial_drop = trial.suggest_float("spatial_dropout", 0.1, 0.5, step=0.1)
    recur_drop   = trial.suggest_float("recurrent_dropout", 0.1, 0.4, step=0.1)
    dense_drop   = trial.suggest_float("dense_dropout", 0.2, 0.6, step=0.1)
    dense_units  = trial.suggest_categorical("dense_units", [32, 64, 128])
    lr           = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [64, 128, 256])
    embed_dim    = trial.suggest_categorical("embed_dim", [50, 100, 200])
    activation   = trial.suggest_categorical("activation", ["relu", "gelu", "tanh"])

    # Build model with this trial's config
    inputs = tf.keras.Input(shape=(128,))
    x = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)(inputs)
    x = tf.keras.layers.SpatialDropout1D(spatial_drop)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units_1, return_sequences=True, recurrent_dropout=recur_drop)
    )(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units_2, recurrent_dropout=recur_drop)
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(dense_units, activation=activation)(x)
    x = tf.keras.layers.Dropout(dense_drop)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Pruning callback — stops unpromising trials early
    pruning_cb = optuna.integration.TFKerasPruningCallback(trial, "val_accuracy")
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)

    history = model.fit(
        X_train_pad, np.array(y_train),
        validation_data=(X_val_pad, np.array(y_val)),
        epochs=15,
        batch_size=batch_size,
        callbacks=[pruning_cb, early_stop],
        verbose=0,
    )

    probs = model.predict(X_val_pad, verbose=0)
    preds = np.argmax(probs, axis=1)
    f1    = f1_score(y_val, preds, average="weighted")

    # Log this trial to MLflow
    with mlflow.start_run(run_name=f"{run_prefix}-trial-{trial.number}", nested=True):
        mlflow.log_params(trial.params)
        mlflow.log_metric("val_f1_weighted", f1)
        mlflow.log_metric("val_accuracy", max(history.history.get("val_accuracy", [0])))

    return f1


def tune(args):
    import tensorflow as tf
    import pickle

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("sentiment-analysis-hpo")

    # Load and prepare data (quick version — no GloVe for HPO speed)
    import pandas as pd
    from ml.models.train_bilstm import build_tokenizer_and_sequences

    df = pd.read_csv(args.data_path, header=None, names=["id", "entity", "sentiment", "text"], dtype=str)
    df = df.dropna(subset=["text", "sentiment"])
    df["text"] = df["text"].apply(clean_for_bilstm)
    df = df[df["text"].str.strip().str.len() > 3]

    le = LabelEncoder()
    labels = le.fit_transform(df["sentiment"])
    texts  = df["text"].tolist()

    # Subsample for speed during HPO (use 40% of data)
    texts, _, labels, _ = train_test_split(texts, labels, test_size=0.6, random_state=42, stratify=labels)

    X_tr, X_vl, y_tr, y_vl = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    tokenizer, X_tr_pad = build_tokenizer_and_sequences(X_tr, max_len=128, vocab_size=30_000)
    X_vl_pad = tf.keras.preprocessing.sequence.pad_sequences(
        tokenizer.texts_to_sequences(X_vl), maxlen=128, padding="post"
    )
    vocab_size  = min(30_000, len(tokenizer.word_index) + 1)
    num_classes = len(le.classes_)

    with mlflow.start_run(run_name=f"bilstm-hpo-{args.n_trials}trials") as parent_run:
        print(f"MLflow parent run: {parent_run.info.run_id}")

        # Create Optuna study with TPE + Hyperband pruning
        sampler = optuna.samplers.TPESampler(seed=42)
        pruner  = optuna.pruners.HyperbandPruner(min_resource=3, max_resource=15, reduction_factor=3)
        study   = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name="bilstm-sentiment-hpo",
        )

        study.optimize(
            lambda trial: objective(trial, X_tr_pad, y_tr, X_vl_pad, y_vl,
                                    vocab_size, num_classes, "bilstm"),
            n_trials=args.n_trials,
            timeout=args.timeout_minutes * 60 if args.timeout_minutes else None,
            show_progress_bar=True,
        )

        # Results
        best = study.best_trial
        print("\n" + "="*60)
        print(f"Best trial #{best.number}  |  F1: {best.value:.4f}")
        print("Best hyperparameters:")
        for k, v in best.params.items():
            print(f"  {k}: {v}")
        print("="*60)

        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        mlflow.log_metric("best_val_f1", best.value)

        # Save best params for reference
        import json
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "best_hparams.json").write_text(json.dumps(best.params, indent=2))
        mlflow.log_artifact(str(output / "best_hparams.json"))

        print(f"\nBest hyperparameters saved to: {output}/best_hparams.json")
        print("Now retrain with: python -m ml.models.train_bilstm --lr <best_lr> --batch_size <best_batch>...")
        return best.params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",         default="twitter_training.csv")
    parser.add_argument("--output_dir",        default="ml/evaluation/hpo")
    parser.add_argument("--n_trials",          type=int, default=30)
    parser.add_argument("--timeout_minutes",   type=int, default=120)
    args = parser.parse_args()
    tune(args)
