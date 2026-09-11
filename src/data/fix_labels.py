"""
Re-label failed golden set entries with rate-limit retry logic.

Usage:
    .venv/bin/python3 src/data/fix_labels.py
"""

import json
import time
from pathlib import Path
from src.config import get_openai_client, PIPELINE_MODEL


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


def main():
    golden_path = "data/golden/golden_set.json"

    with open(golden_path, "r") as f:
        data = json.load(f)

    failed = [i for i, d in enumerate(data) if not d["labels"]["intent"]]
    print(f"Found {len(failed)} failed entries to re-label")

    if not failed:
        print("Nothing to fix!")
        return

    client = get_openai_client()
    fixed = 0
    batch_size = 12  # Stay under 15 req/min limit

    for batch_start in range(0, len(failed), batch_size):
        batch_indices = failed[batch_start : batch_start + batch_size]

        for idx in batch_indices:
            text = data[idx]["customer_text"]
            try:
                response = client.chat.completions.create(
                    model=PIPELINE_MODEL,
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

                data[idx]["labels"] = {
                    "intent": intent,
                    "escalate": bool(parsed.get("escalate", True)),
                    "escalation_reason": parsed.get("escalation_reason", ""),
                    "notes": parsed.get("notes", ""),
                }
                fixed += 1

            except Exception as e:
                print(f"  Failed again on {data[idx]['id']}: {str(e)[:80]}")

        print(f"  Fixed {fixed}/{len(failed)}... waiting 65s for rate limit...")
        if batch_start + batch_size < len(failed):
            time.sleep(65)  # Wait for rate limit window to reset

    # Save
    with open(golden_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Final check
    still_failed = sum(1 for d in data if not d["labels"]["intent"])
    print(f"\n✅ Fixed {fixed} entries. Still failed: {still_failed}")

    # Print updated distribution
    from collections import Counter
    intent_counts = Counter(d["labels"]["intent"] for d in data if d["labels"]["intent"])
    esc_counts = Counter(str(d["labels"]["escalate"]) for d in data)
    print(f"\nIntent Distribution:")
    for intent, count in intent_counts.most_common():
        print(f"  {intent:<30} {count:>4}")
    print(f"\nEscalation Distribution:")
    for esc, count in esc_counts.most_common():
        print(f"  {esc:<10} {count:>4}")


if __name__ == "__main__":
    main()
