"""
Model Evaluation Module
========================
Evaluates trained models and produces:
  - Classification report (accuracy, precision, recall, F1)
  - Confusion matrix
  - Feature importance plot
  - Model comparison chart
  - Per-team prediction accuracy
"""

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/evaluate_model.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

MODELS_DIR    = Path("models")
PROCESSED_DIR = Path("data/processed")
PLOTS_DIR     = Path("models/plots")

sns.set_theme(style="whitegrid", palette="muted")


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_model(name: str):
    path = MODELS_DIR / f"{name}.joblib"
    model = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return model


def load_artifacts() -> tuple:
    le      = joblib.load(MODELS_DIR / "label_encoder.joblib")
    feat_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")
    return le, feat_cols


def load_test_data(feat_cols: list[str]) -> tuple:
    df = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    split = int(len(df) * 0.8)
    test_df = df.iloc[split:]
    available = [c for c in feat_cols if c in test_df.columns]
    X_test = test_df[available]
    le, _ = load_artifacts()
    y_test = le.transform(test_df["result"])
    return X_test, y_test, test_df.iloc[split:].reset_index(drop=True)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(model, X_test: pd.DataFrame, y_test, le) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    target_names = le.classes_.tolist()

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "report":    classification_report(y_test, y_pred, target_names=target_names),
        "cm":        confusion_matrix(y_test, y_pred),
        "y_pred":    y_pred,
        "y_prob":    y_prob,
        "classes":   target_names,
    }
    return metrics


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, classes: list[str], model_name: str):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=14)
    plt.tight_layout()
    path = PLOTS_DIR / f"confusion_matrix_{model_name.lower()}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", path)


def plot_feature_importance(model, feat_cols: list[str], model_name: str, top_n: int = 20):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Walk the pipeline to get the classifier
    clf = model.named_steps.get("clf") if hasattr(model, "named_steps") else model

    if not hasattr(clf, "feature_importances_"):
        logger.warning("%s does not expose feature_importances_.", model_name)
        return

    importances = clf.feature_importances_
    feat_imp = pd.Series(importances, index=feat_cols).sort_values(ascending=False)
    top = feat_imp.head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    top.plot.barh(ax=ax, color="steelblue")
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} Feature Importances – {model_name}", fontsize=14)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = PLOTS_DIR / f"feature_importance_{model_name.lower()}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved feature importance to %s", path)


def plot_model_comparison(comparison: dict[str, dict]):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = ["accuracy", "precision", "recall", "f1"]
    models = list(comparison.keys())
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model_name in enumerate(models):
        vals = [comparison[model_name][m] for m in metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model_name)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14)
    ax.legend()
    plt.tight_layout()
    path = PLOTS_DIR / "model_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved model comparison chart to %s", path)


def plot_team_performance(df_test: pd.DataFrame, y_pred: np.ndarray, le, model_name: str):
    """Bar chart: prediction accuracy per team."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = df_test.copy()
    df["y_pred_label"] = le.inverse_transform(y_pred)
    df["correct"] = df["result"] == df["y_pred_label"]

    all_teams = pd.concat([
        df[["home_team", "correct"]].rename(columns={"home_team": "team"}),
        df[["away_team", "correct"]].rename(columns={"away_team": "team"}),
    ])
    team_acc = all_teams.groupby("team")["correct"].mean().sort_values()

    if team_acc.empty:
        logger.warning("No team accuracy data – skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["coral"] * len(team_acc)
    ax.barh(team_acc.index.tolist(), team_acc.values, color=colors)
    ax.set_title(f"Prediction Accuracy by Team – {model_name}", fontsize=13)
    ax.set_xlabel("Accuracy")
    ax.axvline(float(team_acc.mean()), color="navy", linestyle="--", label="Mean")
    ax.legend()
    plt.tight_layout()
    path = PLOTS_DIR / f"team_accuracy_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved team accuracy chart to %s", path)


# ── Main ───────────────────────────────────────────────────────────────────────

def evaluate(model_names: list[str] | None = None) -> dict:
    """
    Evaluate all trained models and generate plots.

    Returns
    -------
    dict
        Metrics keyed by model name.
    """
    Path("logs").mkdir(exist_ok=True)

    if model_names is None:
        model_names = ["random_forest", "xgboost"]

    le, feat_cols = load_artifacts()
    X_test, y_test, df_test = load_test_data(feat_cols)

    all_metrics = {}

    for name in model_names:
        try:
            model = load_model(name)
        except FileNotFoundError:
            logger.warning("Model '%s' not found, skipping.", name)
            continue

        display_name = name.replace("_", " ").title()
        metrics = compute_metrics(model, X_test, y_test, le)
        all_metrics[display_name] = metrics

        logger.info("\n=== %s ===\n%s", display_name, metrics["report"])
        print(f"\n{'='*60}")
        print(f"  {display_name}")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1-score : {metrics['f1']:.4f}")
        print(f"{'='*60}")
        print(metrics["report"])

        plot_confusion_matrix(metrics["cm"], metrics["classes"], display_name)
        plot_feature_importance(model, feat_cols, display_name)
        plot_team_performance(df_test, metrics["y_pred"], le, display_name)

    if len(all_metrics) >= 1:
        comparison = {
            name: {m: all_metrics[name][m] for m in ["accuracy", "precision", "recall", "f1"]}
            for name in all_metrics
        }
        plot_model_comparison(comparison)

    return all_metrics


if __name__ == "__main__":
    evaluate()
