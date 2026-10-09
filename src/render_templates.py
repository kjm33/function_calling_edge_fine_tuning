"""Neutral dialogs -> per-model chat-template segments (assistant-only loss mask).

Renders a neutral-format conversation (messages + OpenAI-style tools) into a
list of {"text": str, "train": bool} segments whose concatenated texts exactly
reconstruct the model's full chat-template rendering. train=True marks
assistant-generated tokens only (prose AND tool-call blocks); False marks
system/user/tool content and template scaffolding.

Formats (FORMATS):
  qwen3_5      Qwen/Qwen3.5-4B              native template; tool-call arguments
                                            must be dicts (parsed, not JSON
                                            strings); rendered with
                                            enable_thinking=False; <think> block
                                            is template-inserted for assistant
                                            turns after the last user query.
  qwen3_2507   Qwen/Qwen3-4B-Instruct-2507  native hermes template; tool-call
                                            arguments stay JSON strings.
  llama3_json  Llama-3.2-3B-Instruct        native template; arguments must be
                                            dicts (template tojson's them);
                                            only ONE tool call per assistant
                                            message is supported, so multi-call
                                            turns are split into call/result
                                            pairs; tool results render via the
                                            "ipython" header (role=tool works).
  phi4_mini    microsoft/Phi-4-mini-instruct native template; tools are JSON-
                                            dumped into the system message's
                                            "tools" key (flattened {name,
                                            description, parameters}); assistant
                                            tool calls are folded into content
                                            as functools[{...}] (the convention
                                            vLLM's phi4_mini_json parser reads);
                                            tool results use role=tool -> <|tool|>.
  gemma_hermes google/gemma-3-4b-it         gemma has NO native tool support:
                                            native template provides turn
                                            structure (strict user/model
                                            alternation; tool results are folded
                                            into user turns), Hermes-style tool
                                            definitions are injected into the
                                            first user turn inside
                                            <tools>[...]</tools>, assistant tool
                                            calls render as <tool_call>{json}
                                            </tool_call> and tool results as a
                                            following user turn
                                            <tool_result>...</tool_result>.

Importing this module does NOT import transformers (it is imported lazily
inside load_tokenizer), so the module is usable from venvs without it.
"""
from __future__ import annotations

import copy
import json
import os
import pathlib
import re
from functools import lru_cache

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))

FORMATS = {"qwen3_5", "qwen3_2507", "llama3_json", "phi4_mini", "gemma_hermes"}

# (primary, ungated mirror) per format; the mirror is tried when the primary
# raises (gated repo / 401 / 403 / missing).
SOURCES = {
    "qwen3_5": ("Qwen/Qwen3.5-4B", None),
    "qwen3_2507": ("Qwen/Qwen3-4B-Instruct-2507", None),
    "llama3_json": ("meta-llama/Llama-3.2-3B-Instruct", "unsloth/Llama-3.2-3B-Instruct"),
    "phi4_mini": ("microsoft/Phi-4-mini-instruct", None),
    "gemma_hermes": ("google/gemma-3-4b-it", "unsloth/gemma-3-4b-it"),
}

HERMES_TOOL_INSTRUCTIONS = (
    "<tools>\n{tools}\n</tools>\n\n"
    "You have access to the tools above. When you need to use a tool, write a "
    '<tool_call> block with a JSON object: <tool_call>\n{{"name": tool_name, '
    '"arguments": {{...}}}}\n</tool_call> You may write several tool_call blocks. '
    "Tool results are provided in <tool_result> blocks in the following user turn; "
    "wait for them before answering. If no tool is needed, answer normally."
)


# ------------------------------------------------------------------ tokenizers
def _hf_token() -> str | None:
    tok = os.getenv("HF_TOKEN", "").strip()
    if tok:
        return tok
    env = ROOT / ".env"
    if not env.exists():
        env = ROOT.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                tok = line.split("=", 1)[1].strip().strip('"').strip("'")
                if tok:
                    return tok
    return None


@lru_cache(maxsize=None)
def _load_tokenizer_cached(hf_id: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(hf_id, token=_hf_token())


def load_tokenizer(fmt: str, model_id: str | None = None):
    """HF tokenizer for a format (default source, with ungated mirror fallback)."""
    _check_fmt(fmt)
    if model_id:
        return _load_tokenizer_cached(model_id)
    last = None
    for hf_id in SOURCES[fmt]:
        if hf_id is None:
            continue
        try:
            return _load_tokenizer_cached(hf_id)
        except Exception as e:  # gated/401/403/missing -> try mirror
            last = e
    raise RuntimeError(f"no tokenizer loadable for format {fmt}: {last}")


def _check_fmt(fmt: str) -> None:
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {sorted(FORMATS)}")


# --------------------------------------------------------- message preparation
def _parse_args(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _args_to_dicts(messages: list[dict]) -> list[dict]:
    out = copy.deepcopy(messages)
    for msg in out:
        for call in msg.get("tool_calls") or []:
            call["function"]["arguments"] = _parse_args(call["function"].get("arguments"))
    return out


def _llama_view(messages: list[dict]) -> list[dict]:
    """Llama-3.2 renders exactly one JSON call per assistant message: split
    multi-call turns into call/result pairs (results follow in call order)."""
    out: list[dict] = []
    i = 0
    while i < len(messages):
        msg = copy.deepcopy(messages[i])
        for call in msg.get("tool_calls") or []:
            call["function"]["arguments"] = _parse_args(call["function"].get("arguments"))
        calls = msg.get("tool_calls") or []
        if msg.get("role") == "assistant" and len(calls) > 1:
            following = messages[i + 1: i + 1 + len(calls)]
            results = [copy.deepcopy(m) for m in following if m.get("role") == "tool"]
            if len(results) != len(calls):
                raise ValueError(
                    f"llama3_json: assistant turn has {len(calls)} tool_calls "
                    f"but {len(results)} following tool results"
                )
            for call, result in zip(calls, results):
                single = copy.deepcopy(msg)
                single["tool_calls"] = [call]
                out.append(single)
                out.append(result)
            i += 1 + len(calls)
            continue
        out.append(msg)
        i += 1
    return out


def _phi4_view(messages: list[dict], tools: list[dict] | None) -> list[dict]:
    """Phi-4-mini: tools live in the system message as a JSON dump; assistant
    tool calls are folded into content as functools[{...}] (phi4_mini_json)."""
    out = copy.deepcopy(messages)
    for idx, msg in enumerate(out):
        if msg.get("role") == "system" and tools and idx == 0:
            flat = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "parameters": t["function"].get("parameters", {}),
                }
                for t in tools
            ]
            msg["tools"] = json.dumps(flat, ensure_ascii=False)
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            calls = [
                {
                    "name": c["function"]["name"],
                    "arguments": _parse_args(c["function"].get("arguments")),
                }
                for c in msg["tool_calls"]
            ]
            block = "functools" + json.dumps(calls, ensure_ascii=False)
            msg["content"] = ((msg.get("content") or "") + "\n" if msg.get("content") else "") + block
            msg.pop("tool_calls", None)
    return out


def _gemma_view(messages: list[dict], tools: list[dict] | None) -> list[dict]:
    """Gemma needs strict user/model alternation: fold consecutive user-side
    items (tool results, user texts) into single user turns; assistant tool
    calls become <tool_call> blocks inside the model turn; Hermes tool
    definitions are injected into the first user turn."""
    out: list[dict] = []
    pending: list[str] = []

    def flush() -> None:
        if pending:
            out.append({"role": "user", "content": "\n".join(pending)})
            pending.clear()

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            out.append({"role": "system", "content": msg.get("content") or ""})
        elif role == "assistant":
            flush()
            content = msg.get("content") or ""
            if msg.get("tool_calls"):
                blocks = "\n".join(
                    '<tool_call>\n{"name": %s, "arguments": %s}\n</tool_call>'
                    % (
                        json.dumps(c["function"]["name"], ensure_ascii=False),
                        json.dumps(_parse_args(c["function"].get("arguments")), ensure_ascii=False),
                    )
                    for c in msg["tool_calls"]
                )
                content = (content + "\n" if content else "") + blocks
            out.append({"role": "assistant", "content": content})
        elif role == "tool":
            pending.append("<tool_result>\n%s\n</tool_result>" % (msg.get("content") or ""))
        elif role == "user":
            pending.append(msg.get("content") or "")
        else:
            raise ValueError(f"gemma_hermes: unexpected role {role!r}")
    flush()
    if tools:
        tool_block = HERMES_TOOL_INSTRUCTIONS.format(
            tools="\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
        )
        for msg in out:
            if msg["role"] == "user":
                msg["content"] = tool_block + "\n\n" + msg["content"]
                break
    return out


def prepare_messages(fmt: str, messages: list[dict], tools: list[dict] | None) -> tuple[list[dict], list | None]:
    """Per-format message view + tools argument handed to apply_chat_template."""
    _check_fmt(fmt)
    tools = tools or []
    if fmt == "qwen3_5":
        return _args_to_dicts(messages), tools or None
    if fmt == "qwen3_2507":
        return copy.deepcopy(messages), tools or None
    if fmt == "llama3_json":
        return _llama_view(messages), tools or None
    if fmt == "phi4_mini":
        return _phi4_view(messages, tools), tools or None
    return _gemma_view(messages, tools), None


# ------------------------------------------------------------------- rendering
def _apply(tok, view: list[dict], tools_param, fmt: str) -> str:
    kwargs = dict(tools=tools_param, tokenize=False, add_generation_prompt=False)
    if fmt == "qwen3_5":
        kwargs["chat_template_kwargs"] = {"enable_thinking": False}
    return tok.apply_chat_template(view, **kwargs)


def full_render(messages: list[dict], tools: list[dict] | None, fmt: str,
                model_id: str | None = None) -> str:
    """Reference full render of a neutral dialog for a format."""
    tok = load_tokenizer(fmt, model_id)
    view, tools_param = prepare_messages(fmt, messages, tools)
    return _apply(tok, view, tools_param, fmt)


def _segments_from_native(tok, view: list[dict], tools_param, fmt: str) -> list[list]:
    """Incremental prefix-render diff -> per-message segments.

    Qwen3.5 renders assistant turns differently depending on whether a later
    user query exists (<think> block insertion); for messages before the last
    real user turn we render with a sentinel user message appended so the
    think-block decision matches the final render, then strip the sentinel's
    fixed suffix. Template quirks (tool-response regrouping in Qwen templates,
    the trailing eos in the Phi-4-mini template) are repaired via
    common-prefix trimming of the previous segment, preserving its train flag.
    """
    sentinel = None
    last_user = -1
    if fmt == "qwen3_5":
        last_user = max((i for i, m in enumerate(view) if m.get("role") == "user"), default=-1)
        if last_user > 0:
            sentinel = _apply(tok, [{"role": "user", "content": ""}], tools_param, fmt)

    segments: list[list] = []
    prev = ""
    for k, msg in enumerate(view):
        try:
            if sentinel is not None and k < last_user:
                r = _apply(tok, view[:k + 1] + [{"role": "user", "content": ""}], tools_param, fmt)
                core = r[: -len(sentinel)] if r.endswith(sentinel) else r
            else:
                core = _apply(tok, view[:k + 1], tools_param, fmt)
        except Exception:
            # unrenderable prefix (e.g. llama with tools needs a first user
            # message; qwen3.5 needs a user query). Assistant turns must be
            # renderable or the mask cannot be built.
            if msg.get("role") == "assistant":
                raise
            continue
        train = msg.get("role") == "assistant"
        if not prev:
            if core:
                segments.append([core, train])
            prev = core
            continue
        if core.startswith(prev):
            if len(core) > len(prev):
                segments.append([core[len(prev):], train])
        elif core == prev:
            pass  # message contributed nothing (e.g. dropped by the template)
        else:
            cp = 0
            for a, b in zip(prev, core):
                if a != b:
                    break
                cp += 1
            drop = len(prev) - cp
            text, flag = segments[-1]
            if drop > len(text):
                raise ValueError(
                    f"{fmt}: cannot repair message {k} (role={msg.get('role')!r}): "
                    f"regroup overwrites {drop} chars of a {len(text)}-char segment"
                )
            segments[-1] = [text[: len(text) - drop], flag]
            if len(core) > cp:
                segments.append([core[cp:], train])
        prev = core

    segments = [[t, f] for t, f in segments if t]
    rendered = "".join(t for t, _ in segments)
    reference = _apply(tok, view, tools_param, fmt)
    if rendered != reference:
        raise ValueError(
            f"{fmt}: segment reconstruction mismatch "
            f"(got {len(rendered)} chars, reference {len(reference)} chars)"
        )
    return segments


def render_segments(messages: list[dict], tools: list[dict] | None, fmt: str,
                    model_id: str | None = None) -> list[dict]:
    """Neutral dialog -> [{"text": str, "train": bool}] segments (exact
    reconstruction of the format's full chat-template render; train=True only
    on assistant-generated spans)."""
    _check_fmt(fmt)
    tok = load_tokenizer(fmt, model_id)
    view, tools_param = prepare_messages(fmt, messages, tools)
    return [{"text": t, "train": f} for t, f in _segments_from_native(tok, view, tools_param, fmt)]


# ------------------------------------------------------- tool-call round-trips
def _scan_bracket(s: str, start: int) -> int:
    """Index of the ']' matching s[start]=='[' (string-aware)."""
    depth, in_str, esc = 0, False, False
    for j in range(start, len(s)):
        c = s[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return j
    raise ValueError("unbalanced brackets")


def _parse_tool_call_obj(obj: dict) -> dict:
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        args = json.loads(args)
    return {"name": obj["name"], "arguments": args}


def extract_tool_calls(fmt: str, segments: list[dict]) -> list[dict]:
    """Parse assistant tool calls back out of the train segments
    ([{"name", "arguments"(dict)}] in dialog order) — round-trip check."""
    _check_fmt(fmt)
    text = "".join(s["text"] for s in segments if s["train"])
    calls: list[dict] = []
    if fmt in ("qwen3_2507", "gemma_hermes"):
        for m in re.finditer(r"(?s)<tool_call>\s*(.*?)\s*</tool_call>", text):
            calls.append(_parse_tool_call_obj(json.loads(m.group(1))))
    elif fmt == "qwen3_5":
        for m in re.finditer(r"(?s)<tool_call>\n<function=(.*?)>\n(.*?)\n</function>\n</tool_call>", text):
            args: dict = {}
            for pm in re.finditer(r"(?s)<parameter=(.*?)>\n(.*?)\n</parameter>(?:\n|$)", m.group(2)):
                raw = pm.group(2)
                try:
                    val = json.loads(raw)
                except Exception:
                    val = raw
                args[pm.group(1)] = val
            calls.append({"name": m.group(1), "arguments": args})
    elif fmt == "llama3_json":
        for m in re.finditer(
            r"(?s)<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", text
        ):
            body = m.group(1).strip()
            if body.startswith("{"):
                calls.append(_parse_tool_call_obj(json.loads(body)))
    elif fmt == "phi4_mini":
        idx = 0
        while True:
            i = text.find("functools[", idx)
            if i < 0:
                break
            j = _scan_bracket(text, i + len("functools"))
            for obj in json.loads(text[i + len("functools"): j + 1]):
                calls.append(_parse_tool_call_obj(obj))
            idx = j + 1
    return calls
