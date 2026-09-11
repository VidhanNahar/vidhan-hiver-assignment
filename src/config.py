"""
Configuration loader.

Reads settings from .env file and config.yaml.
Supports both OpenAI and Gemini (via OpenAI-compatible endpoint).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (src/config.py → src/ → project root)
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")


def get_env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


# --- API Keys ---
# Support both OpenAI and Gemini
GEMINI_API_KEY = get_env("GEMINI_API_KEY")
OPENAI_API_KEY = get_env("OPENAI_API_KEY") or GEMINI_API_KEY

# --- API Base URL ---
# Gemini's OpenAI-compatible endpoint
API_BASE_URL = get_env("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

# --- Models ---
PIPELINE_MODEL = get_env("PIPELINE_MODEL", "gemini-3.5-flash-lite")
JUDGE_MODEL = get_env("JUDGE_MODEL", "gemini-3.5-flash-lite")

# --- Embedding ---
EMBEDDING_PROVIDER = get_env("EMBEDDING_PROVIDER", "local")

# --- Paths ---
RAW_DATA_PATH = get_env("RAW_DATA_PATH", "data/raw/twcs.csv")
PROCESSED_DATA_PATH = get_env("PROCESSED_DATA_PATH", "data/processed/amazon_threads.json")
GOLDEN_SET_PATH = get_env("GOLDEN_SET_PATH", "data/golden/golden_set.json")
CHROMA_DB_PATH = get_env("CHROMA_DB_PATH", "data/chroma")

# --- RAG ---
RETRIEVER_TOP_K = int(get_env("RETRIEVER_TOP_K", "5"))


def get_openai_client():
    """Create an OpenAI client configured for the current API provider with built-in retry on 429."""
    from openai import OpenAI
    import time
    import re

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=API_BASE_URL,
        timeout=30.0,
    )

    orig_create = client.chat.completions.create

    def retry_create(*args, **kwargs):
        max_retries = 10
        base_delay = 5.0
        for attempt in range(max_retries):
            try:
                # Pacing delay to avoid burst rate-limit errors
                time.sleep(2.0)
                return orig_create(*args, **kwargs)
            except Exception as e:
                err_str = str(e)
                is_rate_limit = (
                    "429" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                    or "rate_limit" in err_str.lower()
                    or "RateLimitError" in type(e).__name__
                )
                if is_rate_limit and attempt < max_retries - 1:
                    delay = base_delay * (1.5 ** attempt)
                    match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
                    if match:
                        delay = float(match.group(1)) + 2.0
                    else:
                        match2 = re.search(r"retryDelay': '(\d+)s", err_str)
                        if match2:
                            delay = float(match2.group(1)) + 2.0
                    delay = max(delay, 5.0)
                    print(f"  [API Rate Limit] Waiting {delay:.1f}s before retry (attempt {attempt+1}/{max_retries})...", flush=True)
                    time.sleep(delay)
                else:
                    raise

    client.chat.completions.create = retry_create
    return client
