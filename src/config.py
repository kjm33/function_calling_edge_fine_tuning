"""Central config: env loading, API keys (dual OpenAI round-robin), paths, model rosters."""
from __future__ import annotations

import os
import pathlib
import random
import threading
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
if not ENV_FILE.exists():
    ENV_FILE = ROOT.parent / ".env"
load_dotenv(ENV_FILE, override=False)

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw_trajectories"
VERIFIED_DIR = DATA_DIR / "verified"
FINAL_DIR = DATA_DIR / "final"
MCP_DIR = ROOT / "mcp"
INSTRUCTION_EXTRACTION_DIR = ROOT.parent / "instruction_extraction"
INSTRUCTIONS_V3 = INSTRUCTION_EXTRACTION_DIR / "data" / "instructions_v3.jsonl"
DIALOGUES = INSTRUCTION_EXTRACTION_DIR / "data" / "dialogues.jsonl"
SCENARIOS = INSTRUCTION_EXTRACTION_DIR / "data" / "scenarios_full.jsonl"

for d in (RAW_DIR, VERIFIED_DIR, FINAL_DIR):
    d.mkdir(parents=True, exist_ok=True)

LANGS = ["en", "de", "pl", "fr", "es"]


def _key(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise SystemExit(f"Missing {name} in {ENV_FILE}")
    return v


HERE_API_KEY = os.getenv("HERE_API_KEY", "").strip()
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
OPENAI_KEYS = [k for k in (os.getenv("OPENAI_API_KEY_1", "").strip(),
                           os.getenv("OPENAI_API_KEY_2", "").strip()) if k]
if not OPENAI_KEYS:
    raise SystemExit("Missing OPENAI_API_KEY_1/2")

# Free-tier provider key pools (all OpenAI-compatible endpoints). Empty = provider unusable.
MISTRAL_KEYS = [k for k in (os.getenv("MISTRAL_API_KEY_1", "").strip(),
                            os.getenv("MISTRAL_API_KEY_2", "").strip()) if k]
GROQ_KEYS = [k for k in (os.getenv("GROQ_API_KEY_1", "").strip(),
                         os.getenv("GROQ_API_KEY_2", "").strip()) if k]
NIM_KEYS = [k for k in (os.getenv("NIM_API_KEY", "").strip(),) if k]

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class _KeyPool:
    """Round-robin over an API-key pool with an in-process dead-key circuit breaker."""

    def __init__(self, keys: list[str]):
        self.keys = keys
        self._idx = random.randrange(len(keys)) if keys else 0
        self._dead: set[str] = set()
        self._lock = threading.Lock()

    def next(self) -> str | None:
        with self._lock:
            if not self.keys:
                return None
            for _ in range(len(self.keys)):
                k = self.keys[self._idx % len(self.keys)]
                self._idx += 1
                if k not in self._dead:
                    return k
            return self.keys[self._idx % len(self.keys)]  # all dead: let the caller fail

    def mark_dead(self, key: str) -> None:
        with self._lock:
            self._dead.add(key)


_openai_pool = _KeyPool(OPENAI_KEYS)
# llm_client.py routes model prefixes "mistral/", "groq/", "nim/" onto these pools.
PROVIDER_POOLS = {
    "mistral/": _KeyPool(MISTRAL_KEYS),
    "groq/": _KeyPool(GROQ_KEYS),
    "nim/": _KeyPool(NIM_KEYS),
}


def mark_dead_key(key: str) -> None:
    """Stop rotating onto an OpenAI key that is permanently out of quota."""
    _openai_pool.mark_dead(key)


def next_openai_key() -> str:
    """Round-robin over both OpenAI accounts (2.5M tok/day minis each), skipping keys
    that ran out of credits for the rest of the process."""
    return _openai_pool.next() or ""


# Teachers: OpenAI minis (primary, cheap+fast) and OpenRouter free roster (diversity).
OPENAI_TEACHERS = ["gpt-5.4-mini", "gpt-5.4-nano", "gpt-4.1-mini", "gpt-4o-mini"]
OPENROUTER_TEACHERS = [
    "qwen/qwen3.5-9b",
    "z-ai/glm-5.3-flash",
    "deepseek/deepseek-v4-flash",
    "google/gemma-4-26b-a4b-it",
]
# Dual independent judges. OpenAI minis removed 10-03: both keys out of credits.
# gemma-4 tested as judge = rubber stamp (avg 9.99, all-10s); glm rejects reasoning:disabled (400).
# deepseek differentiates (spread 7.8-10) and is NOT a gen_main teacher (no self-bias).
# 10-04: OpenRouter key hit monthly limit → default judges are the LOCAL pair (vLLM on this box).
# Override with FCFT_JUDGES="a,b" (e.g. the OpenRouter pair once quota resets).
JUDGE_MODELS = os.getenv("FCFT_JUDGES", "local/qwen3.5-9b,local/gpt-oss-20b").split(",")

# Judge pass threshold (mean of judges, 0-10) for trajectory acceptance.
JUDGE_PASS_SCORE = 7.0

# Student models to fine-tune/compare.
STUDENTS = {
    "qwen3.5-4b": {
        "hf_id": "Qwen/Qwen3.5-4B", "parser": "qwen3_coder", "reasoning_parser": "qwen3",
        "template": "qwen3_coder", "nightly_vllm": True,
    },
    "qwen3-4b-2507": {
        "hf_id": "Qwen/Qwen3-4B-Instruct-2507", "parser": "hermes", "reasoning_parser": None,
        "template": "hermes", "nightly_vllm": False,
    },
    "llama-3.2-3b": {
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct", "parser": "llama3_json",
        "reasoning_parser": None, "template": "llama3_json", "nightly_vllm": False,
    },
    "phi-4-mini": {
        "hf_id": "microsoft/Phi-4-mini-instruct", "parser": "phi4_mini_json",
        "reasoning_parser": None, "template": "phi4_mini", "nightly_vllm": False,
    },
    "gemma-3-4b": {
        "hf_id": "google/gemma-3-4b-it", "parser": "hermes", "reasoning_parser": None,
        "template": "hermes", "nightly_vllm": False, "hermes_transplant": True,
    },
}


def validate_required_keys() -> None:
    missing = [n for n, v in (("HERE_API_KEY", HERE_API_KEY),
                              ("TOMTOM_API_KEY", TOMTOM_API_KEY),
                              ("OPENROUTER_API_KEY", OPENROUTER_API_KEY)) if not v]
    if missing:
        raise SystemExit(f"Missing env keys: {missing}")
