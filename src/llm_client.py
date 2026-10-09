"""Unified teacher LLM client: OpenAI minis (dual-key round-robin) + OpenRouter free roster."""
from __future__ import annotations

import json
import random
import re
import threading
import time
from typing import Any

import httpx

from src import config


class LLMError(RuntimeError):
    pass


class APIStatusError(LLMError):
    """Non-retryable HTTP status from the provider, with body for compat handling."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"{status_code}: {body[:300]}")
        self.status_code = status_code
        self.body = body


# OpenAI reasoning-generation models: reject max_tokens (want max_completion_tokens)
# and reject temperature != default.
_NEW_API_RE = re.compile(r"^(gpt-5|o[0-9])")

# Local vLLM OpenAI-compatible servers (no auth). "local/" prefix routes here.
# A model may have several endpoints (e.g. qwen3.5-9b served on both GPUs);
# requests round-robin across them.
_LOCAL_MODELS: dict[str, list[tuple[str, str]]] = {
    "local/gpt-oss-20b": [("http://127.0.0.1:8000/v1", "gpt-oss-20b")],
    "local/qwen3.5-9b": [("http://127.0.0.1:8001/v1", "qwen3.5-9b")],
}

_local_rr: dict[str, int] = {}
_local_rr_lock = threading.Lock()


def _next_local(model: str) -> tuple[str, str]:
    """Next (base_url, wire_name) endpoint, rotating across replicas."""
    eps = _LOCAL_MODELS[model]
    if len(eps) == 1:
        return eps[0]
    with _local_rr_lock:
        i = _local_rr.get(model, 0)
        _local_rr[model] = (i + 1) % len(eps)
    return eps[i]

# Free-tier external providers (OpenAI-compatible). Prefix -> (base_url, key-pool prefix).
# Checked BEFORE the OpenAI/OpenRouter heuristics below. The prefix is stripped from the
# model id on the wire (e.g. "groq/openai/gpt-oss-120b" -> "openai/gpt-oss-120b").
_PROVIDERS: dict[str, str] = {
    "nim/": "https://integrate.api.nvidia.com/v1",
    "groq/": "https://api.groq.com/openai/v1",
    "mistral/": "https://api.mistral.ai/v1",
}


def _provider_for(model: str) -> tuple[str, str, str] | None:
    """(prefix, base_url, wire_model) if the model routes to an external free provider."""
    for prefix, base in _PROVIDERS.items():
        if model.startswith(prefix):
            return prefix, base, model[len(prefix):]
    return None


def _post(base_url: str, api_key: str, payload: dict, timeout: float = 180.0) -> dict:
    r = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=timeout,
    )
    if r.status_code in (429, 500, 502, 503, 529):
        raise LLMError(f"retryable {r.status_code}: {r.text[:200]}")
    if r.status_code != 200:
        raise APIStatusError(r.status_code, r.text)
    return r.json()


def _adapt_new_api(payload: dict) -> dict:
    """Proactive compat for gpt-5*/o-series: rename max_tokens, drop temperature.

    OpenRouter payloads are passed through untouched (it accepts max_tokens).
    """
    if not _NEW_API_RE.match(payload.get("model", "")):
        return payload
    p = dict(payload)
    if "max_tokens" in p:
        p["max_completion_tokens"] = p.pop("max_tokens")
    p.pop("temperature", None)  # only the default temperature is accepted
    return p


def _adapt_from_400(payload: dict, body: str) -> dict | None:
    """Reactive compat: fix payload per an API 400 complaint. None if not handled."""
    low = body.lower()
    p = dict(payload)
    changed = False
    if "temperature" in low and "temperature" in p:
        p.pop("temperature")
        changed = True
    if re.search(r"max[_ ]tokens", low) and "max_tokens" in p:
        p["max_completion_tokens"] = p.pop("max_tokens")
        changed = True
    if "reasoning_effort" in low and "reasoning_effort" in p:
        p.pop("reasoning_effort")
        changed = True
    return p if changed else None


def chat(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    response_json: bool = False,
    retries: int = 5,
    extra_payload: dict | None = None,
) -> dict:
    """Returns choice.message dict. Rotates OpenAI keys; falls back across retryables.

    extra_payload: optional per-call overrides merged into the OpenRouter payload
    (e.g. {"reasoning": {"enabled": False}} to stop a thinking model from burning
    the whole max_tokens budget before emitting content)."""
    is_local = model in _LOCAL_MODELS
    provider = None if is_local else _provider_for(model)
    is_openai = not provider and not model.startswith(
        ("qwen/", "z-ai/", "deepseek/", "google/", "mistralai/", "openai/gpt-oss", "local/"))
    if provider and not config.PROVIDER_POOLS[provider[0]].keys:
        raise LLMError(f"no API keys configured for {provider[0]} provider")
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
    if response_json:
        payload["response_format"] = {"type": "json_object"}
    last_err: Exception | None = None
    key: str | None = None
    for attempt in range(retries):
        try:
            if is_local:
                base, name = _next_local(model)
                p = dict(payload)
                p["model"] = name
                if "gpt-oss" in name:
                    p["reasoning_effort"] = "low"
                elif "qwen" in name:
                    p["chat_template_kwargs"] = {"enable_thinking": False}
                data = _post(base, "EMPTY", p)
            elif provider:
                _, base, wire_model = provider
                p = dict(payload)
                p["model"] = wire_model
                if "gpt-oss" in wire_model:
                    p["reasoning_effort"] = "low"
                key = config.PROVIDER_POOLS[provider[0]].next() or ""
                data = _post(base, key, p)
            elif is_openai:
                key = config.next_openai_key()
                data = _post(config.OPENAI_BASE_URL, key, _adapt_new_api(payload))
            else:
                extra = {}
                if "glm" in model or "gpt-oss" in model:
                    extra = {"reasoning": {"effort": "low"}}
                elif "qwen3" in model:
                    extra = {"reasoning": {"effort": "low", "exclude": True}}
                if extra_payload:
                    extra.update(extra_payload)
                data = _post(config.OPENROUTER_BASE_URL, config.OPENROUTER_API_KEY, {**payload, **extra})
            msg = data["choices"][0]["message"]
            if msg.get("reasoning_content") and not msg.get("content"):
                msg["content"] = msg["reasoning_content"]  # thinking-only fallback
            return msg
        except APIStatusError as e:
            if provider and e.status_code == 401 and key:
                # Bad/revoked key: rotate it out of the pool and retry on another key.
                config.PROVIDER_POOLS[provider[0]].mark_dead(key)
                last_err = e
                continue
            # 400 caused by a known param incompatibility: fix payload, retry at once.
            if e.status_code == 400 and (is_openai or is_local or provider):
                adapted = _adapt_from_400(payload, e.body)
                if adapted is not None:
                    payload = adapted
                    last_err = e
                    continue
            raise  # permanent client error: retrying won't help
        except (LLMError, httpx.HTTPError, KeyError) as e:
            if is_openai and key and "insufficient_quota" in str(e):
                config.mark_dead_key(key)  # out of credits: never rotate onto it again
            last_err = e
            time.sleep(min(2 ** attempt + random.random(), 30))
    raise LLMError(f"{model} failed after {retries} attempts: {last_err}")


def chat_json(model: str, messages: list[dict], **kw: Any) -> dict:
    """Chat expecting a JSON object response (parses fenced or raw JSON)."""
    msg = chat(model, messages, response_json=True, **kw)
    return parse_json_blob(msg.get("content") or "")


def parse_json_blob(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass
    for delim in ("```json", "```"):
        if delim in text:
            inner = text.split(delim, 1)[1].split("```", 1)[0]
            try:
                return json.loads(inner)
            except Exception:
                continue
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start >= 0:
        try:
            return json.loads(text[start:text.rfind("}") + 1] if text[start] == "{" else text[start:text.rfind("]") + 1])
        except Exception:
            pass
    raise LLMError(f"unparseable JSON: {text[:200]}")
