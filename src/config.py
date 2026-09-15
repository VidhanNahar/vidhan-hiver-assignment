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

# --- Load declarative config.yaml if present ---
CONFIG_YAML_PATH = _project_root / "configs" / "config.yaml"
_yaml_config = {}
if CONFIG_YAML_PATH.exists():
    try:
        import yaml
        with open(CONFIG_YAML_PATH, "r", encoding="utf-8") as f:
            _yaml_config = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[Config] Warning: Could not load {CONFIG_YAML_PATH}: {e}")


def get_config_val(key: str, env_key: str = None, default: str = "") -> str:
    """Get config with precedence: Environment Var > .env > config.yaml > default."""
    env_name = env_key or key.upper()
    env_val = os.getenv(env_name)
    if env_val is not None and env_val != "":
        return env_val
    if key in _yaml_config:
        return str(_yaml_config[key])
    return default


def get_env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def parse_bool(val, default: bool = True) -> bool:
    """Safely parse boolean values without treating non-empty string 'false' as True."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        cleaned = val.strip().lower()
        if cleaned in ("true", "1", "yes", "t", "y"):
            return True
        if cleaned in ("false", "0", "no", "f", "n"):
            return False
    return default


# --- API Keys ---
# Support both OpenAI and Gemini
GEMINI_API_KEY = get_config_val("gemini_api_key", "GEMINI_API_KEY", "")
OPENAI_API_KEY = get_config_val("openai_api_key", "OPENAI_API_KEY", "") or GEMINI_API_KEY

# --- API Base URL ---
# Gemini's OpenAI-compatible endpoint default
API_BASE_URL = get_config_val(
    "api_base_url",
    "API_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/" if GEMINI_API_KEY and not os.getenv("OPENAI_API_KEY")
    else "https://api.openai.com/v1",
)

# --- Models ---
# Separate default judge model from pipeline model for independent evaluation
_is_gemini = "generativelanguage.googleapis.com" in API_BASE_URL or (bool(GEMINI_API_KEY) and not os.getenv("OPENAI_API_KEY"))
_default_pipeline = "gemini-2.5-flash" if _is_gemini else "gpt-4o-mini"
_default_judge = "gemini-2.5-pro" if _is_gemini else "gpt-4o"

PIPELINE_MODEL = get_config_val("pipeline_model", "PIPELINE_MODEL", _default_pipeline)
JUDGE_MODEL = get_config_val("judge_model", "JUDGE_MODEL", _default_judge)

# --- Embedding ---
EMBEDDING_PROVIDER = get_config_val("embedding_provider", "EMBEDDING_PROVIDER", "local")

# --- Paths ---
RAW_DATA_PATH = get_config_val("raw_data_path", "RAW_DATA_PATH", "data/raw/twcs.csv")
PROCESSED_DATA_PATH = get_config_val("processed_data_path", "PROCESSED_DATA_PATH", "data/processed/amazon_threads.json")
SAMPLE_FIXTURE_PATH = get_config_val("sample_fixture_path", "SAMPLE_FIXTURE_PATH", "data/fixtures/sample_threads.json")
GOLDEN_SET_PATH = get_config_val("golden_set_path", "GOLDEN_SET_PATH", "data/golden/golden_set.json")
CHROMA_DB_PATH = get_config_val("chroma_db_path", "CHROMA_DB_PATH", "data/chroma")

# --- RAG ---
RETRIEVER_TOP_K = int(get_config_val("retriever_top_k", "RETRIEVER_TOP_K", "5"))
RETRIEVER_INDEX_SIZE = int(get_config_val("retriever_index_size", "RETRIEVER_INDEX_SIZE", "10000"))

# --- Escalation ---
LOW_CONFIDENCE_THRESHOLD = float(
    _yaml_config.get("escalation", {}).get("low_confidence_threshold", 0.4)
    if isinstance(_yaml_config.get("escalation"), dict)
    else 0.4
)

# --- Rate Limit / Pacing ---
# Set API_PACING_DELAY in seconds (e.g. 1.0 or 2.0) if using restricted free tier RPM. Default is 0.0 (no delay).
API_PACING_DELAY = float(get_env("API_PACING_DELAY", "0.0"))


def get_openai_client(api_key: str = None, base_url: str = None):
    """Create an OpenAI client configured for the current API provider with built-in retry on 429."""
    from openai import OpenAI
    import time
    import re

    key = api_key or OPENAI_API_KEY
    url = base_url or API_BASE_URL

    client = OpenAI(
        api_key=key,
        base_url=url,
        timeout=30.0,
    )

    orig_create = client.chat.completions.create

    def retry_create(*args, **kwargs):
        max_retries = 10
        base_delay = 5.0
        for attempt in range(max_retries):
            try:
                # Optional pacing delay only if configured
                if API_PACING_DELAY > 0:
                    time.sleep(API_PACING_DELAY)
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
