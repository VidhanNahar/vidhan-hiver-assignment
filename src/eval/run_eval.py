"""
Evaluation Runner — Unified entry point for running the full evaluation.

Runs a system (main, trivial baseline, or simple baseline) against
the golden evaluation set and produces a comprehensive report.

Usage:
    python3 -m src.eval.run_eval --system main
    python3 -m src.eval.run_eval --system baseline_trivial
    python3 -m src.eval.run_eval --system baseline_simple
    python3 -m src.eval.run_eval --system all
"""

import argparse
import json
import time
from pathlib import Path

from src.config import GOLDEN_SET_PATH
from src.eval.metrics import (
    compute_classification_metrics,
    compute_escalation_metrics,
    format_classification_report,
    format_escalation_report,
)
from src.eval.llm_judge import LLMJudge, format_judge_report


def load_golden_set(path: str = None) -> list[dict]:
    """Load the golden evaluation set."""
    path = path or GOLDEN_SET_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_main_system(golden: list[dict], skip_judge: bool = False) -> dict:
    """Run the main pipeline on the golden set."""
    from src.pipeline.agent import SupportAgent

    print("\n[Eval] Initializing main system...")
    agent = SupportAgent()

    print(f"[Eval] Processing {len(golden)} examples with main system...")
    results = []
    start = time.time()

    for i, example in enumerate(golden, 1):
        print(f"  [{i}/{len(golden)}] {example['id']} ({example['labels']['intent'][:20]})...", flush=True)

        response = agent.process(example["customer_text"])
        results.append({
            "id": example["id"],
            "customer_text": example["customer_text"],
            "true_intent": example["labels"]["intent"],
            "pred_intent": response.intent,
            "intent_confidence": response.intent_confidence,
            "true_escalate": example["labels"]["escalate"],
            "pred_escalate": response.escalate,
            "escalation_reason": response.escalation_reason,
            "draft_reply": response.draft_reply,
            "historical_reply": example.get("historical_brand_reply", ""),
            "processing_time_ms": response.processing_time_ms,
        })

    elapsed = time.time() - start
    print(f"[Eval] Main system: {len(results)} examples in {elapsed:.1f}s")

    return _compute_all_metrics(results, "main", skip_judge=skip_judge)


def run_trivial_baseline(golden: list[dict], skip_judge: bool = False) -> dict:
    """Run the trivial baseline on the golden set."""
    from src.baselines.trivial import TrivialBaseline

    print("\n[Eval] Running trivial baseline...")
    baseline = TrivialBaseline()

    results = []
    for example in golden:
        response = baseline.process(example["customer_text"])
        results.append({
            "id": example["id"],
            "customer_text": example["customer_text"],
            "true_intent": example["labels"]["intent"],
            "pred_intent": response.intent,
            "intent_confidence": 1.0,
            "true_escalate": example["labels"]["escalate"],
            "pred_escalate": response.escalate,
            "escalation_reason": response.escalation_reason,
            "draft_reply": response.draft_reply,
            "historical_reply": example.get("historical_brand_reply", ""),
            "processing_time_ms": 0,
        })

    return _compute_all_metrics(results, "baseline_trivial", skip_judge=skip_judge)


def run_simple_baseline(golden: list[dict], skip_judge: bool = False) -> dict:
    """Run the simple (zero-shot LLM) baseline on the golden set."""
    from src.baselines.simple import SimpleBaseline

    print("\n[Eval] Running simple baseline...")
    baseline = SimpleBaseline()

    results = []
    for i, example in enumerate(golden, 1):
        print(f"  [{i}/{len(golden)}] {example['id']} ({example['labels']['intent'][:20]})...", flush=True)

        response = baseline.process(example["customer_text"])
        results.append({
            "id": example["id"],
            "customer_text": example["customer_text"],
            "true_intent": example["labels"]["intent"],
            "pred_intent": response.intent,
            "intent_confidence": response.intent_confidence,
            "true_escalate": example["labels"]["escalate"],
            "pred_escalate": response.escalate,
            "escalation_reason": response.escalation_reason,
            "draft_reply": response.draft_reply,
            "historical_reply": example.get("historical_brand_reply", ""),
            "processing_time_ms": 0,
        })

    return _compute_all_metrics(results, "baseline_simple", skip_judge=skip_judge)


def _compute_all_metrics(results: list[dict], system_name: str, skip_judge: bool = False) -> dict:
    """Compute classification, escalation, and judge metrics."""
    # Intent classification
    y_true_intent = [r["true_intent"] for r in results]
    y_pred_intent = [r["pred_intent"] for r in results]
    intent_metrics = compute_classification_metrics(y_true_intent, y_pred_intent)

    print(format_classification_report(intent_metrics))

    # Escalation (only if labels exist)
    y_true_esc = [r["true_escalate"] for r in results if r["true_escalate"] is not None]
    y_pred_esc = [r["pred_escalate"] for r in results if r["true_escalate"] is not None]

    esc_metrics = {}
    if y_true_esc:
        esc_metrics = compute_escalation_metrics(y_true_esc, y_pred_esc)
        print(format_escalation_report(esc_metrics))

    # LLM Judge (optional — expensive)
    judge_scores = []
    if not skip_judge:
        print(f"[Eval] Running LLM judge on {len(results)} replies...")
        judge = LLMJudge()
        judge_inputs = [
            {
                "customer_text": r["customer_text"],
                "draft_reply": r["draft_reply"],
                "historical_reply": r.get("historical_reply", ""),
                "intent": r["pred_intent"],
            }
            for r in results
        ]
        judge_scores_obj = judge.score_batch(judge_inputs, verbose=True)
        judge_scores = [s.to_dict() for s in judge_scores_obj]
        print(format_judge_report(judge_scores_obj))

    # Combine everything
    output = {
        "system": system_name,
        "num_examples": len(results),
        "intent_metrics": intent_metrics,
        "escalation_metrics": esc_metrics,
        "judge_scores": judge_scores,
        "detailed_results": results,
    }

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"eval_{system_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[Eval] Results saved to {out_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument(
        "--system",
        choices=["main", "baseline_trivial", "baseline_simple", "all"],
        default="main",
        help="Which system to evaluate",
    )
    parser.add_argument(
        "--golden",
        default=None,
        help="Path to golden set JSON (default: from config)",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM judge scoring (saves API cost)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to the first N examples",
    )
    args = parser.parse_args()

    golden = load_golden_set(args.golden)

    # Filter to only labelled examples
    labelled = [g for g in golden if g["labels"]["intent"]]
    if len(labelled) < len(golden):
        print(f"[Eval] Warning: only {len(labelled)}/{len(golden)} examples have labels. Using labelled subset.")
    golden = labelled

    if args.limit and args.limit > 0:
        print(f"[Eval] Subsetting evaluation to first {args.limit} examples.")
        golden = golden[:args.limit]

    if not golden:
        print("[Eval] ERROR: No labelled examples found in golden set. Label them first!")
        return

    runners = {
        "main": run_main_system,
        "baseline_trivial": run_trivial_baseline,
        "baseline_simple": run_simple_baseline,
    }

    if args.system == "all":
        all_results = {}
        for name, runner in runners.items():
            all_results[name] = runner(golden, skip_judge=args.skip_judge)

        # Print comparison table
        print("\n" + "=" * 70)
        print("  COMPARISON TABLE")
        print("=" * 70)
        print(f"  {'System':<20} {'Macro-F1':>10} {'Accuracy':>10} {'Esc Recall':>12}")
        print("-" * 70)
        for name, res in all_results.items():
            mf1 = res["intent_metrics"]["macro_f1"]
            acc = res["intent_metrics"]["accuracy"]
            esc_rec = res["escalation_metrics"].get("recall", "N/A")
            if isinstance(esc_rec, float):
                esc_rec = f"{esc_rec:.2%}"
            print(f"  {name:<20} {mf1:>10.2%} {acc:>10.2%} {esc_rec:>12}")
        print("=" * 70)
    else:
        runners[args.system](golden, skip_judge=args.skip_judge)


if __name__ == "__main__":
    main()
