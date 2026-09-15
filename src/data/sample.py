"""
Stratified Sampling for Golden Evaluation Set

Samples ~200 examples from the processed AmazonHelp threads using
a stratified + adversarial strategy:
  - 70% stratified random (proportional across intent buckets)
  - 20% hard/ambiguous cases (short, long, multi-intent signals)
  - 10% edge cases (very short, emoji-heavy, non-English)

Usage:
    python3 -m src.data.sample
    python3 -m src.data.sample --n 200 --out data/golden/golden_candidates.json
"""

import argparse
import json
import random
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Heuristic intent detection (for stratified sampling — NOT the real classifier)
# ---------------------------------------------------------------------------

INTENT_KEYWORDS = {
    "Order Status / Tracking": [
        "where is my order", "tracking", "when will", "order status",
        "hasn't arrived", "hasn't shipped", "still processing", "estimated delivery",
        "dispatch", "shipping update", "not shipped", "order update",
    ],
    "Refund / Return": [
        "refund", "money back", "return", "reimburse", "credit back",
        "charged", "overcharged", "wrong charge", "want my money",
    ],
    "Delivery Problem": [
        "not delivered", "lost package", "wrong address", "stolen",
        "damaged", "broken", "missing package", "says delivered",
        "never received", "delivery failed", "wrong item",
    ],
    "Account / Login Issue": [
        "can't log in", "login", "password", "locked out", "hacked",
        "account blocked", "suspended", "can't access", "two factor",
        "verification", "account closed",
    ],
    "Product / Service Question": [
        "how do i", "how to", "does it", "compatible", "difference between",
        "which one", "recommend", "what is", "can i use",
    ],
    "Prime / Subscription": [
        "prime", "subscription", "membership", "prime video", "prime music",
        "annual plan", "free trial", "cancel prime", "prime day",
    ],
    "Complaint / Frustration": [
        "worst", "terrible", "horrible", "disgusting", "never again",
        "unacceptable", "ridiculous", "pathetic", "angry", "furious",
        "disappointed", "frustrat", "waste of time", "awful",
    ],
}


def heuristic_intent(text: str) -> str:
    """Simple keyword-based intent detection for sampling stratification."""
    text_lower = text.lower()
    scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[intent] = score

    if not scores:
        return "Other / Miscellaneous"

    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Edge case detection
# ---------------------------------------------------------------------------

RE_NON_ASCII = re.compile(r"[^\x00-\x7F]")
RE_EMOJI = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]",
    flags=re.UNICODE,
)
RE_PHONE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")

NON_ENGLISH_SIGNALS = [
    "por favor", "gracias", "merci", "bitte", "danke", "bonjour",
    "hola", "obrigado", "nicht", "keine", "vous", "nous",
    "comprei", "recebi", "minha", "também", "já",
]


def is_edge_case(thread: dict) -> bool:
    """Check if a thread is an edge case worth including."""
    text = thread["customer_text"]

    # Very short messages
    if len(text) < 15:
        return True

    # Very long messages
    if len(text) > 250:
        return True

    # Heavy emoji usage (count individual emoji characters)
    if len(RE_EMOJI.findall(text)) >= 3:
        return True

    # Non-English
    text_lower = text.lower()
    if any(sig in text_lower for sig in NON_ENGLISH_SIGNALS):
        return True

    # Contains potential PII patterns (order numbers, emails, phone numbers)
    if re.search(r"\d{3}-\d{7}-\d{7}", text):  # Amazon order ID pattern
        return True
    if re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text):
        return True
    if RE_PHONE.search(text):
        return True

    return False


def is_hard_case(thread: dict) -> bool:
    """Check if a message is ambiguous / multi-intent."""
    text = thread["customer_text"].lower()

    # Multiple intents detected
    intents_matched = sum(
        1 for keywords in INTENT_KEYWORDS.values()
        if any(kw in text for kw in keywords)
    )
    if intents_matched >= 2:
        return True

    # Questions phrased as complaints
    if ("?" in text) and any(w in text for w in ["worst", "terrible", "why", "how come"]):
        return True

    return False


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_golden_set(
    threads: list[dict],
    n: int = 200,
    seed: int = 42,
) -> list[dict]:
    """Sample n examples using stratified + adversarial strategy."""
    random.seed(seed)

    # Classify all threads by heuristic intent
    by_intent = {}
    for t in threads:
        intent = heuristic_intent(t["customer_text"])
        by_intent.setdefault(intent, []).append(t)

    print("=== Heuristic Intent Distribution (full dataset) ===")
    for intent, items in sorted(by_intent.items(), key=lambda x: -len(x[1])):
        pct = len(items) / len(threads) * 100
        print(f"  {intent:<30} {len(items):>7,} ({pct:5.1f}%)")

    # Separate edge cases and hard cases
    edge_cases = [t for t in threads if is_edge_case(t)]
    hard_cases = [t for t in threads if is_hard_case(t) and not is_edge_case(t)]

    print(f"\n  Edge cases found: {len(edge_cases):,}")
    print(f"  Hard cases found: {len(hard_cases):,}")

    # Allocate counts
    n_edge = min(int(n * 0.10), len(edge_cases))      # 10% edge
    n_hard = min(int(n * 0.20), len(hard_cases))       # 20% hard
    n_stratified = n - n_edge - n_hard                  # 70% stratified

    # Sample edge cases
    sampled_edge = random.sample(edge_cases, n_edge)
    sampled_edge_ids = {t["thread_id"] for t in sampled_edge}

    # Sample hard cases (excluding already-sampled edges)
    hard_available = [t for t in hard_cases if t["thread_id"] not in sampled_edge_ids]
    sampled_hard = random.sample(hard_available, min(n_hard, len(hard_available)))
    sampled_hard_ids = {t["thread_id"] for t in sampled_hard}

    # Stratified sample from remaining
    already_sampled = sampled_edge_ids | sampled_hard_ids
    remaining_by_intent = {}
    for intent, items in by_intent.items():
        remaining = [t for t in items if t["thread_id"] not in already_sampled]
        if remaining:
            remaining_by_intent[intent] = remaining

    # Proportional allocation with strict minimum floor guarantee (5 per intent)
    quotas = {}
    for intent, items in remaining_by_intent.items():
        quotas[intent] = min(5, len(items))

    floor_allocated = sum(quotas.values())
    remaining_budget = max(0, n_stratified - floor_allocated)

    remaining_pool_weights = {
        intent: max(0, len(items) - quotas[intent])
        for intent, items in remaining_by_intent.items()
    }
    total_pool_weight = sum(remaining_pool_weights.values())

    if total_pool_weight > 0 and remaining_budget > 0:
        extras = {}
        for intent, weight in remaining_pool_weights.items():
            extras[intent] = int(remaining_budget * (weight / total_pool_weight))

        remainder = remaining_budget - sum(extras.values())
        sorted_by_weight = sorted(
            remaining_pool_weights.keys(),
            key=lambda k: (remaining_pool_weights[k], k),
            reverse=True,
        )
        for i in range(remainder):
            extras[sorted_by_weight[i % len(sorted_by_weight)]] += 1

        for intent in quotas:
            quotas[intent] += extras.get(intent, 0)

    sampled_stratified = []
    for intent, items in remaining_by_intent.items():
        k = min(quotas[intent], len(items))
        sampled_stratified.extend(random.sample(items, k))

    # Combine
    all_sampled = sampled_edge + sampled_hard + sampled_stratified
    random.shuffle(all_sampled)

    # Format for labelling
    golden = []
    for i, t in enumerate(all_sampled):
        golden.append({
            "id": f"golden_{i+1:03d}",
            "customer_text": t["customer_text"],
            "customer_text_raw": t["customer_text_raw"],
            "historical_brand_reply": t["brand_reply"],
            "thread_id": t["thread_id"],
            "heuristic_intent": heuristic_intent(t["customer_text"]),
            "sampling_category": (
                "edge_case" if t["thread_id"] in sampled_edge_ids
                else "hard_case" if t["thread_id"] in sampled_hard_ids
                else "stratified"
            ),
            "labels": {
                "intent": "",
                "escalate": None,
                "escalation_reason": "",
                "notes": "",
            },
        })

    print(f"\n=== Sampled Golden Set ===")
    print(f"  Stratified: {len(sampled_stratified)}")
    print(f"  Hard cases: {len(sampled_hard)}")
    print(f"  Edge cases: {len(sampled_edge)}")
    print(f"  Total:      {len(golden)}")

    # Distribution of heuristic intents in sample
    print(f"\n=== Heuristic Intent Distribution (golden set) ===")
    from collections import Counter
    intent_counts = Counter(g["heuristic_intent"] for g in golden)
    for intent, count in intent_counts.most_common():
        print(f"  {intent:<30} {count:>4}")

    return golden


def main():
    parser = argparse.ArgumentParser(description="Sample golden evaluation set")
    parser.add_argument("--n", type=int, default=200, help="Number of examples to sample")
    parser.add_argument(
        "--input", default="data/processed/amazon_threads.json",
        help="Path to processed threads"
    )
    parser.add_argument(
        "--out", default="data/golden/golden_candidates.json",
        help="Output path for golden candidates"
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        threads = json.load(f)

    golden = sample_golden_set(threads, n=args.n)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(golden)} candidates to {args.out}")
    print("Next step: hand-label the 'labels' field in each example.")


if __name__ == "__main__":
    main()
