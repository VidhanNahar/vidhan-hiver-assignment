"""
Quick Exploratory Analysis of processed AmazonHelp threads.

Prints stats and samples to help design the intent taxonomy.

Usage:
    python3 -m src.data.explore
"""

import json
import random
from collections import Counter
from pathlib import Path


def load_threads(path: str = "data/processed/amazon_threads.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def explore(threads: list[dict]):
    print(f"Total threads: {len(threads):,}\n")

    # --- Customer text length distribution ---
    lens = [len(t["customer_text"]) for t in threads]
    print("=== Customer Text Length Distribution ===")
    buckets = [(0, 20), (21, 50), (51, 100), (101, 150), (151, 200), (201, 286)]
    for lo, hi in buckets:
        count = sum(1 for l in lens if lo <= l <= hi)
        pct = count / len(lens) * 100
        bar = "█" * int(pct / 2)
        print(f"  {lo:>3}-{hi:>3} chars: {count:>6,} ({pct:5.1f}%) {bar}")

    # --- Most common words in customer messages ---
    print("\n=== Top 30 Words in Customer Messages ===")
    word_counter = Counter()
    stop_words = {"i", "the", "a", "to", "and", "is", "in", "it", "my", "of",
                  "for", "you", "me", "on", "that", "this", "was", "have", "with",
                  "but", "be", "are", "not", "so", "at", "do", "if", "has", "just",
                  "been", "no", "from", "can", "an", "they", "your", "will", "all",
                  "would", "there", "or", "what", "about", "up", "out", "get",
                  "when", "one", "had", "am", "got", "url", "don't", "i'm", "it's",
                  "we", "how", "as", "their", "them", "than", "by", "its", "he",
                  "she", "did", "also", "any", "even", "im", "ive", "cant", "dont",
                  "now", "still", "after", "over", "way", "too", "very", "back",
                  "some", "going", "more", "could", "being", "were", "who", "re",
                  "which", "should", "why", "here", "because", "other", "each"}
    for t in threads:
        words = t["customer_text"].lower().split()
        for w in words:
            w = w.strip(".,!?;:'\"()-[]{}#@")
            if len(w) > 2 and w not in stop_words:
                word_counter[w] += 1

    for word, count in word_counter.most_common(30):
        print(f"  {word:<20} {count:>6,}")

    # --- Sample 30 random customer messages ---
    print("\n=== 30 Random Customer Messages ===")
    sample = random.sample(threads, min(30, len(threads)))
    for i, t in enumerate(sample, 1):
        text = t["customer_text"][:120]
        reply = t["brand_reply"][:80]
        print(f"\n  [{i:>2}] CUSTOMER: {text}")
        print(f"       AMAZON:   {reply}")

    # --- Thread depth distribution ---
    print("\n\n=== Thread Depth Distribution ===")
    depths = [len(t["full_thread"]) for t in threads]
    depth_counter = Counter(depths)
    for depth in sorted(depth_counter.keys())[:10]:
        count = depth_counter[depth]
        pct = count / len(depths) * 100
        bar = "█" * int(pct / 2)
        print(f"  {depth:>2} messages: {count:>6,} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    random.seed(42)
    threads = load_threads()
    explore(threads)
