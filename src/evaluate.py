"""
evaluate.py
-----------
Evaluation metrics, ROC curve, confusion matrix, and
feature importance plotting utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    classification_report, confusion_matrix, roc_curve,
)

PALETTE = {"stayed": "#1E3A5F", "churned": "#E84855"}


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Return AUC-ROC, F1, and Accuracy as a dict."""
    return {
        "auc_roc":  round(float(roc_auc_score(y_true, y_proba)), 4),
        "f1":       round(float(f1_score(y_true, y_pred)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
    }


def print_metrics(model_name: str, metrics: dict) -> None:
    print(f"\n{'─'*50}")
    print(f"  {model_name}")
    print(f"{'─'*50}")
    print(f"  AUC-ROC  : {metrics['auc_roc']:.4f}")
    print(f"  F1 Score : {metrics['f1']:.4f}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")


# ── Plots ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8F9FA",
    "axes.grid":        True,
    "grid.color":       "white",
    "grid.linewidth":   1.2,
    "font.size":        11,
})


def plot_roc_curve(
    models: dict,      # {"Model Name": (y_true, y_proba)}
    save_path: str | Path | None = None,
) -> None:
    """
    Plot ROC curves for one or more models.

    Parameters
    ----------
    models    : dict mapping model name → (y_true, y_proba)
    save_path : if provided, save figure to this path
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#93C5FD", "#E84855", "#10B981"]

    for (name, (y_true, y_proba)), color in zip(models.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str = "Model",
    save_path: str | Path | None = None,
) -> None:
    """Plot a labelled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Stayed", "Churned"],
        yticklabels=["Stayed", "Churned"],
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def plot_feature_importance(
    feature_importances,
    feature_names: list,
    top_n: int = 15,
    save_path: str | Path | None = None,
) -> None:
    """
    Horizontal bar chart of top-N feature importances
    with plain-English labels where available.
    """
    PLAIN_LABELS = {
        "change_mou":           "Usage trend\n(min change vs 3-mo avg)",
        "change_rev":           "Revenue trend\n(rev change vs 3-mo avg)",
        "eqpdays":              "Device age\n(days on current handset)",
        "rev_per_min":          "Revenue efficiency\n($ per minute of use)",
        "totmrc_Mean":          "Monthly bill\n(recurring charge)",
        "months":               "Tenure\n(months as customer)",
        "mou_Mean":             "Monthly usage\n(minutes of use)",
        "call_completion_rate": "Call quality\n(% calls completed)",
        "drop_rate":            "Drop rate\n(% calls dropped)",
        "overage_flag":         "Overage flag\n(any overage incurred)",
    }

    import pandas as pd
    fi = (
        pd.Series(feature_importances, index=feature_names)
        .sort_values(ascending=False)
        .head(top_n)
    )

    labels = [PLAIN_LABELS.get(f, f) for f in fi.index]
    colors = (
        ["#E84855"] * 3 +
        ["#F59E0B"] * 2 +
        ["#1E3A5F"] * (top_n - 5)
    )

    fig, ax = plt.subplots(figsize=(9, top_n * 0.55))
    ax.barh(range(top_n), fi.values[::-1], color=colors[::-1], alpha=0.9)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels[::-1], fontsize=10)
    ax.set_xticks([])
    ax.set_xlabel("Relative Predictive Importance", fontsize=11)
    ax.set_title(
        f"Top {top_n} Signals the Model Uses to Predict Churn",
        fontsize=13, fontweight="bold", pad=10,
    )
    for i, v in enumerate(fi.values[::-1]):
        ax.text(v + 2, i, f"#{top_n - i}", va="center", fontsize=9, color="#334155")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def plot_risk_tiers(
    df_scored,          # dataframe with churn_prob and churn columns
    save_path: str | Path | None = None,
) -> None:
    """Bar charts for customer count and revenue at risk per tier."""
    import pandas as pd

    df_scored = df_scored.copy()
    df_scored["risk_tier"] = pd.cut(
        df_scored["churn_prob"],
        bins=[0, 0.4, 0.6, 1.0],
        labels=["Low\n(<40%)", "Medium\n(40–60%)", "High\n(>60%)"],
    )

    risk = df_scored.groupby("risk_tier", observed=True).agg(
        n              = ("churn_prob", "count"),
        churn_rate     = ("churn",      "mean"),
        avg_rev        = ("rev_Mean",   "mean"),
    ).reset_index()
    risk["rev_at_risk_M"] = risk["n"] * risk["churn_rate"] * risk["avg_rev"] * 12 / 1e6

    tier_colors = ["#86EFAC", "#FDE68A", "#FCA5A5"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    b1 = axes[0].bar(risk["risk_tier"].astype(str), risk["n"], color=tier_colors)
    for bar, row in zip(b1, risk.itertuples()):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{row.n:,}", ha="center", fontsize=11, fontweight="bold",
        )
    axes[0].set_title("Customers by Risk Tier", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Customers")

    b2 = axes[1].bar(risk["risk_tier"].astype(str), risk["rev_at_risk_M"], color=tier_colors)
    for bar, row in zip(b2, risk.itertuples()):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"${row.rev_at_risk_M:.1f}M", ha="center", fontsize=11, fontweight="bold",
        )
    axes[1].set_title("Annual Revenue at Risk by Tier ($M)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Revenue ($M)")
    axes[1].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:.0f}M")
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()
