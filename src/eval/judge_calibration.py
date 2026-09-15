"""
Judge Calibration — Human vs LLM judge agreement analysis.

Computes Pearson correlation, Cohen's κ, and MAE between
human and LLM judge scores to validate the evaluation rubric.
"""

import json
import math
from pathlib import Path
from typing import Optional


def pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient between two lists."""
    n = len(x)
    if n < 2:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

    if std_x == 0 or std_y == 0:
        return 0.0

    return round(cov / (std_x * std_y), 4)


def mean_absolute_error(x: list[float], y: list[float]) -> float:
    """Compute Mean Absolute Error between two lists."""
    if not x:
        return 0.0
    return round(sum(abs(xi - yi) for xi, yi in zip(x, y)) / len(x), 4)


def cohens_kappa(x: list[int], y: list[int], num_classes: int = 3) -> float:
    """Compute Cohen's κ for ordinal agreement.

    Bins scores into Low (1-2), Medium (3), High (4-5) before computing.
    """
    def bin_score(s: int) -> int:
        if s <= 2:
            return 0  # Low
        elif s == 3:
            return 1  # Medium
        else:
            return 2  # High

    x_binned = [bin_score(xi) for xi in x]
    y_binned = [bin_score(yi) for yi in y]

    n = len(x_binned)
    if n == 0:
        return 0.0

    # Observed agreement
    p_o = sum(1 for xi, yi in zip(x_binned, y_binned) if xi == yi) / n

    # Expected agreement by chance
    from collections import Counter
    x_counts = Counter(x_binned)
    y_counts = Counter(y_binned)

    p_e = sum(
        (x_counts.get(k, 0) / n) * (y_counts.get(k, 0) / n)
        for k in range(num_classes)
    )

    if p_e == 1.0:
        return 1.0

    return round((p_o - p_e) / (1.0 - p_e), 4)


DIMENSIONS = ["relevance", "groundedness", "tone", "actionability", "safety"]


def calibrate(
    human_scores: list[dict],
    llm_scores: list[dict],
) -> dict:
    """Compute agreement metrics between human and LLM judge scores.

    Args:
        human_scores: List of dicts with keys: relevance, groundedness, tone, actionability, safety (1-5)
        llm_scores: List of dicts with same keys (1-5)

    Returns:
        Dict with per-dimension Pearson r, Cohen's κ, MAE, and overall summary.
    """
    if not human_scores or not llm_scores:
        raise ValueError("Score lists must not be empty")
    if len(human_scores) != len(llm_scores):
        raise ValueError(
            f"Score lists must be same length: got {len(human_scores)} human scores and {len(llm_scores)} LLM scores"
        )

    results = {}

    for dim in DIMENSIONS:
        h = [s[dim] for s in human_scores]
        l = [s[dim] for s in llm_scores]

        results[dim] = {
            "pearson_r": pearson_correlation(
                [float(x) for x in h], [float(x) for x in l]
            ),
            "cohens_kappa": cohens_kappa(h, l),
            "mae": mean_absolute_error(
                [float(x) for x in h], [float(x) for x in l]
            ),
            "human_mean": round(sum(h) / len(h), 2),
            "llm_mean": round(sum(l) / len(l), 2),
        }

    # Overall
    all_h = [s[dim] for s in human_scores for dim in DIMENSIONS]
    all_l = [s[dim] for s in llm_scores for dim in DIMENSIONS]

    results["overall"] = {
        "pearson_r": pearson_correlation(
            [float(x) for x in all_h], [float(x) for x in all_l]
        ),
        "mae": mean_absolute_error(
            [float(x) for x in all_h], [float(x) for x in all_l]
        ),
    }

    return results


def format_calibration_report(results: dict) -> str:
    """Format calibration results as a report."""
    lines = [
        "",
        "=" * 70,
        "  Judge Calibration: Human vs LLM Agreement",
        "=" * 70,
        f"  {'Dimension':<18} {'Pearson r':>10} {'Cohen κ':>10} {'MAE':>8} {'H_mean':>8} {'L_mean':>8}",
        "-" * 70,
    ]

    for dim in DIMENSIONS:
        d = results[dim]
        lines.append(
            f"  {dim.capitalize():<18} {d['pearson_r']:>10.3f} {d['cohens_kappa']:>10.3f} "
            f"{d['mae']:>8.3f} {d['human_mean']:>8.2f} {d['llm_mean']:>8.2f}"
        )

    lines.append("-" * 70)
    o = results["overall"]
    lines.append(f"  {'OVERALL':<18} {o['pearson_r']:>10.3f} {'—':>10} {o['mae']:>8.3f}")
    lines.append("=" * 70)

    # Interpretation
    r = o["pearson_r"]
    if r >= 0.8:
        interp = "Strong agreement — LLM judge is well-calibrated."
    elif r >= 0.6:
        interp = "Moderate agreement — LLM judge is usable but has blind spots."
    elif r >= 0.4:
        interp = "Weak agreement — LLM judge scores should be interpreted with caution."
    else:
        interp = "Poor agreement — LLM judge needs recalibration or is unreliable."

    lines.append(f"\n  Interpretation: {interp}")
    lines.append("")

    return "\n".join(lines)


def main():
    """CLI runner for judge calibration."""
    import argparse
    parser = argparse.ArgumentParser(description="Run Judge Calibration (Human vs LLM)")
    parser.add_argument(
        "--input",
        default="data/golden/calibration_scores.json",
        help="Path to calibration dataset JSON with paired human and llm scores",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[Error] Calibration file not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    human_scores = [d["human"] for d in data]
    llm_scores = [d["llm"] for d in data]

    results = calibrate(human_scores, llm_scores)
    print(format_calibration_report(results))

    out_path = Path("results/judge_calibration.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Calibration results saved to {out_path}")


if __name__ == "__main__":
    main()
