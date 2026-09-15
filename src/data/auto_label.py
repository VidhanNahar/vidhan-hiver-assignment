"""
Auto-labeller for the golden evaluation set.

Uses the LLM to pre-fill intent and escalation labels on golden candidates.
These MUST be reviewed and corrected by hand — this is a time-saver, not a replacement.

Usage:
    python3 -m src.data.auto_label
    python3 -m src.data.auto_label --input data/golden/golden_candidates.json --output data/golden/golden_set.json
"""

import argparse
import json
import time
from pathlib import Path

from openai import OpenAI

from src.config import OPENAI_API_KEY, PIPELINE_MODEL, get_openai_client


LABELLING_PROMPT = """You are labelling customer support tweets for evaluation.

For each customer message, provide:
1. **intent** — the primary intent from this list:
   - Order Status / Tracking
   - Refund / Return
   - Delivery Problem
   - Account / Login Issue
   - Product / Service Question
   - Prime / Subscription
   - Complaint / Frustration
   - Other / Miscellaneous

2. **escalate** — true/false. Escalate if:
   - Requires account verification or private info
   - Customer requests money/refund
   - PII is shared publicly
   - Legal threats, safety concerns
   - Extremely angry or complex multi-step issue
   Do NOT escalate for simple questions, thanks/praise, or easily answered queries.

3. **escalation_reason** — brief explanation for the escalation decision.

4. **notes** — any observations (multi-intent, non-English, ambiguous, etc.)

OUTPUT FORMAT (JSON only):
{"intent": "...", "escalate": true/false, "escalation_reason": "...", "notes": "..."}"""


VALID_INTENTS = [
    "Order Status / Tracking", "Refund / Return", "Delivery Problem",
    "Account / Login Issue", "Product / Service Question", "Prime / Subscription",
    "Complaint / Frustration", "Other / Miscellaneous",
]


def auto_label(candidates: list[dict], api_key: str = None, model: str = None) -> list[dict]:
    """Auto-label golden candidates using LLM."""
    client = get_openai_client(api_key=api_key)
    model = model or PIPELINE_MODEL

    labelled = []
    total = len(candidates)

    for i, candidate in enumerate(candidates, 1):
        text = candidate["customer_text"]

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": LABELLING_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)

            intent = parsed.get("intent", "Other / Miscellaneous")
            if intent not in VALID_INTENTS:
                intent = "Other / Miscellaneous"

            raw_esc = parsed.get("escalate")
            if isinstance(raw_esc, bool):
                esc_val = raw_esc
            elif isinstance(raw_esc, str) and raw_esc.strip().lower() in ("true", "false"):
                esc_val = raw_esc.strip().lower() == "true"
            else:
                esc_val = None  # Leave malformed results for manual review

            candidate["labels"] = {
                "intent": intent,
                "escalate": esc_val,
                "escalation_reason": parsed.get("escalation_reason", ""),
                "notes": parsed.get("notes", "") if esc_val is not None else "NEEDS MANUAL REVIEW (malformed escalate value)",
            }

        except Exception as e:
            candidate["labels"] = {
                "intent": "",
                "escalate": None,
                "escalation_reason": f"Auto-labelling failed: {str(e)}",
                "notes": "NEEDS MANUAL LABEL",
            }

        labelled.append(candidate)

        if i % 20 == 0 or i == total:
            print(f"  Labelled {i}/{total}...")

    return labelled


def main():
    parser = argparse.ArgumentParser(description="Auto-label golden candidates")
    parser.add_argument(
        "--input", default="data/golden/golden_candidates.json",
        help="Path to golden candidates",
    )
    parser.add_argument(
        "--output", default="data/golden/golden_set.json",
        help="Output path for labelled golden set",
    )
    args = parser.parse_args()

    print(f"Loading candidates from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    print(f"Auto-labelling {len(candidates)} examples...")
    labelled = auto_label(candidates)

    # Stats
    from collections import Counter
    intent_counts = Counter(c["labels"]["intent"] for c in labelled if c["labels"]["intent"])
    esc_counts = Counter(str(c["labels"]["escalate"]) for c in labelled)

    print(f"\n=== Auto-Label Summary ===")
    print(f"  Total: {len(labelled)}")
    print(f"\n  Intent Distribution:")
    for intent, count in intent_counts.most_common():
        print(f"    {intent:<30} {count:>4}")
    print(f"\n  Escalation Distribution:")
    for esc, count in esc_counts.most_common():
        print(f"    {esc:<10} {count:>4}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labelled, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {args.output}")
    print(f"\n⚠️  IMPORTANT: These labels are LLM-generated. You MUST review and correct them manually!")
    print(f"   Open {args.output} and verify each label, especially edge cases and ambiguous ones.")


if __name__ == "__main__":
    main()
