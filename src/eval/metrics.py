"""
Automated Metrics — F1, Precision, Recall, Confusion Matrix.

Computes classification and escalation metrics against the golden set.
"""

import json
from collections import Counter
from typing import Optional


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: Optional[list[str]] = None,
) -> dict:
    """Compute per-class and macro classification metrics.

    Args:
        y_true: Ground truth intent labels.
        y_pred: Predicted intent labels.
        labels: Optional list of all label names.

    Returns:
        Dict with per-class P/R/F1, macro averages, accuracy, and confusion matrix.
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    # Per-class metrics
    per_class = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(1 for t in y_true if t == label)

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    # Macro averages
    classes_with_support = [c for c in labels if per_class[c]["support"] > 0]
    macro_precision = sum(per_class[c]["precision"] for c in classes_with_support) / max(len(classes_with_support), 1)
    macro_recall = sum(per_class[c]["recall"] for c in classes_with_support) / max(len(classes_with_support), 1)
    macro_f1 = sum(per_class[c]["f1"] for c in classes_with_support) / max(len(classes_with_support), 1)

    # Accuracy
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(len(y_true), 1)

    # Confusion matrix
    confusion = {}
    for label_true in labels:
        confusion[label_true] = {}
        for label_pred in labels:
            confusion[label_true][label_pred] = sum(
                1 for t, p in zip(y_true, y_pred) if t == label_true and p == label_pred
            )

    return {
        "per_class": per_class,
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(accuracy, 4),
        "confusion_matrix": confusion,
        "total_samples": len(y_true),
    }


def compute_escalation_metrics(
    y_true: list[bool],
    y_pred: list[bool],
) -> dict:
    """Compute escalation decision metrics.

    Focuses on False Negative Rate since a missed escalation
    is worse than a false alarm.
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(len(y_true), 1)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "false_negative_rate": round(fnr, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "total_samples": len(y_true),
    }


def format_classification_report(metrics: dict) -> str:
    """Format classification metrics as a human-readable table."""
    lines = [
        "",
        "=" * 75,
        f"  Intent Classification Report  (n={metrics['total_samples']})",
        "=" * 75,
        f"  {'Intent':<30} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}",
        "-" * 75,
    ]
    for label, m in sorted(metrics["per_class"].items()):
        lines.append(
            f"  {label:<30} {m['precision']:>6.2%} {m['recall']:>6.2%} {m['f1']:>6.2%} {m['support']:>8}"
        )
    lines.append("-" * 75)
    lines.append(
        f"  {'MACRO AVERAGE':<30} {metrics['macro_precision']:>6.2%} {metrics['macro_recall']:>6.2%} {metrics['macro_f1']:>6.2%} {metrics['total_samples']:>8}"
    )
    lines.append(f"  Accuracy: {metrics['accuracy']:.2%}")
    lines.append("=" * 75)
    return "\n".join(lines)


def format_escalation_report(metrics: dict) -> str:
    """Format escalation metrics as a human-readable table."""
    lines = [
        "",
        "=" * 55,
        f"  Escalation Decision Report  (n={metrics['total_samples']})",
        "=" * 55,
        f"  Precision:            {metrics['precision']:.2%}",
        f"  Recall:               {metrics['recall']:.2%}",
        f"  F1:                   {metrics['f1']:.2%}",
        f"  Accuracy:             {metrics['accuracy']:.2%}",
        f"  False Negative Rate:  {metrics['false_negative_rate']:.2%}  ← critical",
        "",
        f"  TP={metrics['true_positives']}  FP={metrics['false_positives']}  "
        f"FN={metrics['false_negatives']}  TN={metrics['true_negatives']}",
        "=" * 55,
    ]
    return "\n".join(lines)
