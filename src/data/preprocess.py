"""
Data Preprocessing Pipeline for AmazonHelp Tweets

Reads the raw TWCS dataset, filters to AmazonHelp threads,
reconstructs customer→brand reply pairs, cleans text, and outputs
structured JSON for downstream use.

Usage:
    python -m src.data.preprocess
    python -m src.data.preprocess --raw data/raw/twcs.csv --out data/processed/amazon_threads.json
"""

import argparse
import csv
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

# Pattern: @mention at the very start of a tweet (possibly multiple)
RE_LEADING_MENTIONS = re.compile(r"^(?:@\w+\s*)+")

# Pattern: URLs
RE_URL = re.compile(r"https?://\S+")

# Pattern: multiple whitespace
RE_MULTI_SPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Clean a single tweet's text.

    Steps:
      1. Decode HTML entities (&amp; → &, etc.)
      2. Remove leading @mentions (brand/customer handles at start)
      3. Replace URLs with [URL] placeholder
      4. Collapse multiple whitespace into single space
      5. Strip leading/trailing whitespace
    """
    # HTML entities
    text = html.unescape(text)

    # Remove leading @mentions
    text = RE_LEADING_MENTIONS.sub("", text)

    # Replace URLs
    text = RE_URL.sub("[URL]", text)

    # Collapse whitespace
    text = RE_MULTI_SPACE.sub(" ", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Thread Reconstruction
# ---------------------------------------------------------------------------

BRAND_AUTHOR_ID = "AmazonHelp"


def load_raw_csv(path: str) -> list[dict]:
    """Load the raw TWCS CSV and return all rows as dicts."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def filter_amazon_threads(rows: list[dict]) -> tuple[dict, dict]:
    """Separate AmazonHelp tweets and inbound customer tweets that are
    part of AmazonHelp conversations.

    Returns:
        brand_tweets: dict mapping tweet_id → row (AmazonHelp outbound)
        all_tweets:   dict mapping tweet_id → row (all tweets for lookup)
    """
    brand_tweets = {}
    all_tweets = {}

    for row in rows:
        tid = row.get("tweet_id", "").strip()
        if not tid:
            continue
        all_tweets[tid] = row

        author = row.get("author_id", "").strip()
        inbound = row.get("inbound", "").strip().lower()

        if author == BRAND_AUTHOR_ID and inbound == "false":
            brand_tweets[tid] = row

    return brand_tweets, all_tweets


def reconstruct_threads(
    brand_tweets: dict, all_tweets: dict
) -> list[dict]:
    """Build customer→brand reply pairs.

    For each AmazonHelp outbound tweet, find the inbound customer tweet
    it responds to. Build structured thread records.
    """
    threads = []
    seen_pairs = set()

    for brand_tid, brand_row in brand_tweets.items():
        # Find what this brand tweet is responding to
        responds_to = brand_row.get("in_response_to_tweet_id", "").strip()
        if not responds_to:
            continue

        # Look up the customer tweet
        customer_row = all_tweets.get(responds_to)
        if customer_row is None:
            continue

        # Verify it's an inbound tweet
        if customer_row.get("inbound", "").strip().lower() != "true":
            continue

        # Avoid duplicate pairs
        pair_key = (responds_to, brand_tid)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        customer_text_raw = customer_row.get("text", "").strip()
        brand_text_raw = brand_row.get("text", "").strip()

        if not customer_text_raw or not brand_text_raw:
            continue

        customer_text_clean = clean_text(customer_text_raw)
        brand_text_clean = clean_text(brand_text_raw)

        # Skip if cleaning left us with empty text
        if not customer_text_clean or not brand_text_clean:
            continue

        # Build the full thread (best-effort: follow the chain)
        full_thread = _build_full_thread(responds_to, all_tweets)

        thread = {
            "thread_id": f"{responds_to}__{brand_tid}",
            "customer_tweet_id": responds_to,
            "brand_tweet_id": brand_tid,
            "customer_text": customer_text_clean,
            "customer_text_raw": customer_text_raw,
            "brand_reply": brand_text_clean,
            "brand_reply_raw": brand_text_raw,
            "full_thread": full_thread,
            "customer_created_at": customer_row.get("created_at", ""),
            "brand_created_at": brand_row.get("created_at", ""),
        }
        threads.append(thread)

    return threads


def _build_full_thread(start_tweet_id: str, all_tweets: dict, max_depth: int = 10) -> list[dict]:
    """Follow the conversation chain from a starting tweet.

    Walks both directions: backwards (in_response_to) and forwards
    (response_tweet_id) to build the full thread.
    """
    thread_msgs = []
    visited = set()

    # Walk backwards from start tweet
    current_id = start_tweet_id
    backward_chain = []
    depth = 0
    while current_id and depth < max_depth:
        if current_id in visited:
            break
        visited.add(current_id)
        tweet = all_tweets.get(current_id)
        if tweet is None:
            break
        backward_chain.append(tweet)
        current_id = tweet.get("in_response_to_tweet_id", "").strip()
        depth += 1

    backward_chain.reverse()
    thread_msgs.extend(backward_chain)

    # Walk forwards from start tweet via response_tweet_id
    current_id = start_tweet_id
    tweet = all_tweets.get(current_id)
    if tweet:
        response_ids = tweet.get("response_tweet_id", "").strip()
        if response_ids:
            for rid in response_ids.split(","):
                rid = rid.strip()
                if rid and rid not in visited:
                    visited.add(rid)
                    resp_tweet = all_tweets.get(rid)
                    if resp_tweet:
                        thread_msgs.append(resp_tweet)

    # Format for output
    formatted = []
    for msg in thread_msgs:
        author = msg.get("author_id", "unknown")
        inbound = msg.get("inbound", "").strip().lower() == "true"
        text = clean_text(msg.get("text", ""))
        formatted.append({
            "author": author,
            "role": "customer" if inbound else "brand",
            "text": text,
        })

    return formatted


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(threads: list[dict]) -> list[dict]:
    """Remove near-duplicate threads based on exact customer_text match."""
    seen_texts = set()
    unique = []
    for t in threads:
        key = t["customer_text"].lower().strip()
        if key not in seen_texts:
            seen_texts.add(key)
            unique.append(t)
    return unique


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(threads: list[dict]) -> None:
    """Print summary statistics about the processed dataset."""
    print(f"\n{'='*60}")
    print(f"  AmazonHelp Preprocessing Summary")
    print(f"{'='*60}")
    print(f"  Total thread pairs:         {len(threads):,}")

    # Customer text length stats
    cust_lens = [len(t["customer_text"]) for t in threads]
    print(f"  Customer text length (chars):")
    print(f"    min: {min(cust_lens)},  median: {sorted(cust_lens)[len(cust_lens)//2]},  max: {max(cust_lens)}")

    # Brand reply length stats
    brand_lens = [len(t["brand_reply"]) for t in threads]
    print(f"  Brand reply length (chars):")
    print(f"    min: {min(brand_lens)},  median: {sorted(brand_lens)[len(brand_lens)//2]},  max: {max(brand_lens)}")

    # Thread depth stats
    thread_lens = [len(t["full_thread"]) for t in threads]
    print(f"  Thread depth (messages):")
    print(f"    min: {min(thread_lens)},  median: {sorted(thread_lens)[len(thread_lens)//2]},  max: {max(thread_lens)}")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def preprocess(raw_path: str, output_path: str) -> list[dict]:
    """Run the full preprocessing pipeline."""
    print(f"[1/5] Loading raw CSV from {raw_path}...")
    rows = load_raw_csv(raw_path)
    print(f"       Loaded {len(rows):,} total rows.")

    print(f"[2/5] Filtering AmazonHelp tweets...")
    brand_tweets, all_tweets = filter_amazon_threads(rows)
    print(f"       Found {len(brand_tweets):,} AmazonHelp outbound tweets.")

    print(f"[3/5] Reconstructing customer→brand thread pairs...")
    threads = reconstruct_threads(brand_tweets, all_tweets)
    print(f"       Reconstructed {len(threads):,} thread pairs.")

    print(f"[4/5] Deduplicating...")
    threads = deduplicate(threads)
    print(f"       {len(threads):,} unique thread pairs after dedup.")

    print(f"[5/5] Saving to {output_path}...")
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(threads, f, indent=2, ensure_ascii=False)
    print(f"       Saved successfully.")

    print_stats(threads)

    return threads


def main():
    parser = argparse.ArgumentParser(description="Preprocess TWCS dataset for AmazonHelp")
    parser.add_argument(
        "--raw", default="data/raw/twcs.csv",
        help="Path to raw twcs.csv (default: data/raw/twcs.csv)"
    )
    parser.add_argument(
        "--out", default="data/processed/amazon_threads.json",
        help="Output path for processed JSON (default: data/processed/amazon_threads.json)"
    )
    args = parser.parse_args()

    preprocess(args.raw, args.out)


if __name__ == "__main__":
    main()
