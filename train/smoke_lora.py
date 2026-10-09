"""LoRA fine-tuning smoke test for function-calling candidates.

Loads a model on ONE GPU, wraps it in LoRA (r=32, alpha=64, dropout=0.05,
all-linear targets, bf16), and runs N training steps on a tiny hardcoded
synthetic tool-call dataset (system -> user -> assistant tool_calls ->
tool result -> assistant final). Loss is computed on assistant tokens only
via a manual label mask built from incremental chat-template renders.

Usage:
    python smoke_lora.py --model Qwen/Qwen3-4B-Instruct-2507 --device cuda:0
    python smoke_lora.py --model Qwen/Qwen3.5-4B --device cuda:1
"""

import argparse
import random
import sys
import time

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "geocode",
            "description": "Resolve a free-form address or place name to coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "Address or place name"},
                },
                "required": ["address"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_route",
            "description": "Compute a driving route between two coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "object",
                        "properties": {"lat": {"type": "number"}, "lng": {"type": "number"}},
                        "required": ["lat", "lng"],
                    },
                    "destination": {
                        "type": "object",
                        "properties": {"lat": {"type": "number"}, "lng": {"type": "number"}},
                        "required": ["lat", "lng"],
                    },
                    "avoid_tolls": {"type": "boolean"},
                },
                "required": ["origin", "destination"],
            },
        },
    },
]

SYSTEM = (
    "You are a navigation assistant with access to map tools. "
    "Use the provided tools when they help answer the user."
)


def sample(addr, place, lat, lng, route_to, route_xy, dist_s, dur_s, two_calls=False):
    args = {"address": addr}
    calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "geocode", "arguments": _json_dump(args)},
        }
    ]
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Where is {place}?"},
        {"role": "assistant", "content": "", "tool_calls": calls},
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": _json_dump({"items": [{"title": place, "lat": lat, "lng": lng}]}),
        },
        {
            "role": "assistant",
            "content": f"{place} is at latitude {lat}, longitude {lng}.",
        },
    ]
    if two_calls:
        msgs[2]["tool_calls"].append(
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "get_route",
                    "arguments": _json_dump(
                        {
                            "origin": {"lat": route_xy[0], "lng": route_xy[1]},
                            "destination": {"lat": lat, "lng": lng},
                        }
                    ),
                },
            }
        )
        msgs.insert(
            4,
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "content": _json_dump({"distance_km": dist_s, "duration_min": dur_s}),
            },
        )
        msgs[-1]["content"] += (
            f" The drive from the origin is {dist_s} km and takes about {dur_s} minutes."
        )
    return {"messages": msgs, "tools": TOOLS}


def _json_dump(obj):
    import json

    return json.dumps(obj, ensure_ascii=False)


def build_samples():
    return [
        sample("Brandenburger Tor, Berlin", "Brandenburger Tor", 52.5163, 13.3777, None, (52.52, 13.405), "2.4", "9", two_calls=True),
        sample("Krakow Main Square", "Rynek Glowny", 50.0616, 19.9373, None, (50.0647, 19.945), "1.1", "5"),
        sample("Eiffel Tower, Paris", "Eiffel Tower", 48.8584, 2.2945, None, (48.8566, 2.3522), "5.3", "18"),
        sample("Sagrada Familia, Barcelona", "Sagrada Familia", 41.4036, 2.1744, None, (41.3874, 2.1686), "2.7", "12"),
        sample("Rijksmuseum, Amsterdam", "Rijksmuseum", 52.3600, 4.8852, None, (52.3702, 4.8952), "1.8", "8"),
        sample("Charles Bridge, Prague", "Charles Bridge", 50.0865, 14.4114, None, (50.0875, 14.4213), "0.9", "4"),
        sample("Golden Gate Bridge, San Francisco", "Golden Gate Bridge", 37.8199, -122.4783, None, (37.7749, -122.4194), "12.6", "25"),
        sample("Tokyo Tower, Tokyo", "Tokyo Tower", 35.6586, 139.7454, None, (35.6812, 139.7671), "6.4", "20"),
    ]


def _args_to_dicts(messages):
    """Deep-copy messages, converting tool_call arguments JSON strings to dicts
    (some templates, e.g. Qwen3.5, iterate arguments as a mapping)."""
    import copy
    import json

    out = copy.deepcopy(messages)
    for msg in out:
        for call in msg.get("tool_calls") or []:
            args = call["function"].get("arguments")
            if isinstance(args, str):
                call["function"]["arguments"] = json.loads(args)
    return out


def render_and_mask(tokenizer, example):
    """Render a tool-call conversation and build labels = loss on assistant turns only.

    Renders incrementally, message by message; the diff between consecutive
    prefix renders isolates each turn, and every assistant turn (tool_calls
    turns and the final answer) is marked for the loss. Tool/user/system turns
    are masked. Handles template quirks: Qwen3.5 requires arguments as a
    mapping and refuses system-only prefixes; both Qwen templates regroup
    consecutive tool responses into a single turn (repaired via common-prefix
    diff, which is safe because it only affects masked tool turns).
    Falls back to final-turn-only or full-sequence loss if the template is
    still not reconstructable.
    """
    messages, tools = example["messages"], example["tools"]

    def build(messages, mode):
        def rend(msgs):
            return tokenizer.apply_chat_template(
                msgs, tools=tools, tokenize=False, add_generation_prompt=False
            )

        segments = []
        prev = ""
        started = False
        repaired = False
        for k, msg in enumerate(messages, start=1):
            try:
                r = rend(messages[:k])
            except Exception:
                if msg["role"] == "assistant":
                    return None, mode
                continue  # unrenderable prefix (e.g. system-only for Qwen3.5)
            if not started:
                segments.append((r, msg["role"] == "assistant"))
            elif r.startswith(prev):
                segments.append((r[len(prev):], msg["role"] == "assistant"))
            else:
                # template regrouped earlier turns (consecutive tool responses):
                # trim the now-invalid tail of the previous segment and mask the
                # diff from the last common prefix
                if msg["role"] == "assistant":
                    return None, mode
                cp = 0
                for a, b in zip(prev, r):
                    if a != b:
                        break
                    cp += 1
                drop = len(prev) - cp
                last_text = segments[-1][0]
                if drop > len(last_text):
                    return None, mode
                segments[-1] = (last_text[: len(last_text) - drop], False)
                segments.append((r[cp:], False))
                repaired = True
            prev = r
            started = True
        if not started:
            return None, mode
        if "".join(text for text, _ in segments) != rend(messages):
            return None, mode
        return segments, mode + ("(tool-regroup-repair)" if repaired else "")

    segments, mode = build(messages, "assistant-only")
    if segments is None:
        segments, mode = build(_args_to_dicts(messages), "assistant-only(dict-args)")
    if segments is None:
        # fallback: mask everything except the final assistant turn
        try:
            r_prev = rend_safe(tokenizer, messages[:-1], tools)
            r_full = rend_safe(tokenizer, messages, tools)
        except Exception:
            r_prev, r_full = None, None
        if r_prev is not None and r_full is not None and r_full.startswith(r_prev):
            mode = "final-turn-only"
            segments = [(r_prev, False), (r_full[len(r_prev):], True)]
        else:
            mode = "full-sequence"
            segments = [(rend_safe(tokenizer, messages, tools), True)]

    ids, labels = [], []
    for text, train_on in segments:
        seg_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.extend(seg_ids)
        labels.extend(seg_ids if train_on else [-100] * len(seg_ids))

    if len(ids) > 2048:
        ids, labels = ids[:2048], labels[:2048]
    n_train = sum(1 for x in labels if x != -100)
    return ids, labels, mode, n_train


def rend_safe(tokenizer, messages, tools):
    try:
        return tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return tokenizer.apply_chat_template(
            _args_to_dicts(messages), tools=tools, tokenize=False, add_generation_prompt=False
        )


def load_model_and_tokenizer(model_id, device, attn):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    kwargs = dict(dtype=torch.bfloat16, device_map={"": device}, attn_implementation=attn)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception as e:
        print(f"[load] AutoModelForCausalLM failed ({type(e).__name__}: {str(e)[:160]}); "
              "trying AutoModelForImageTextToText", flush=True)
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.config.use_cache = False
    return model, tokenizer


def apply_lora(model, target_modules="all-linear"):
    cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    return get_peft_model(model, cfg)


def naive_linear_targets(model):
    """Fallback: every nn.Linear leaf name except lm_head."""
    names = set()
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            leaf = name.split(".")[-1]
            if leaf != "lm_head":
                names.add(leaf)
    return sorted(names)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.attn)
    print(f"[load] class={type(model).__name__} model_type={model.config.model_type} "
          f"in {time.time()-t0:.1f}s", flush=True)

    samples = build_samples()
    dataset = []
    mask_modes = set()
    for ex in samples:
        ids, labels, mode, n_train = render_and_mask(tokenizer, ex)
        if len(ids) > args.seq_len:
            print(f"[data] sample exceeds seq-len ({len(ids)} > {args.seq_len}) — skipping", flush=True)
            continue
        mask_modes.add(mode)
        dataset.append((ids, labels, n_train))
    print(f"[data] {len(dataset)} samples, mask mode(s)={sorted(mask_modes)}, "
          f"trainable tokens/sample={[d[2] for d in dataset]}", flush=True)
    if not dataset:
        sys.exit("FATAL: no usable samples")

    # LoRA
    try:
        model = apply_lora(model)
        lora_mode = "all-linear"
    except Exception as e:
        print(f"[lora] all-linear failed ({type(e).__name__}: {str(e)[:200]}); "
              "falling back to naive nn.Linear target list", flush=True)
        targets = naive_linear_targets(model)
        print(f"[lora] naive targets: {targets}", flush=True)
        model = apply_lora(model, target_modules=targets)
        lora_mode = "naive-list"
    trainable, total = model.get_nb_trainable_parameters()
    print(f"[lora] mode={lora_mode} trainable={trainable/1e6:.1f}M / {total/1e9:.2f}B "
          f"({100*trainable/total:.2f}%)", flush=True)

    # gradient checkpointing
    gc_status = "on"
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    except Exception as e:
        gc_status = f"off (failed: {type(e).__name__})"
        print(f"[gc] gradient checkpointing unavailable: {e}", flush=True)

    model.train()
    optimizer = torch.optim.AdamW(
        [p_ for p_ in model.parameters() if p_.requires_grad], lr=args.lr
    )

    # fixed sample order (wrapping) so repeat visits of the same sample give a
    # like-for-like learning signal
    n = len(dataset)
    visits = {}
    print("[train] step | sample | loss | lr", flush=True)
    for step in range(args.steps):
        i = step % n
        ids, labels, _ = dataset[i]
        input_ids = torch.tensor([ids], device=args.device)
        lbl = torch.tensor([labels], device=args.device)
        out = model(input_ids=input_ids, labels=lbl)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p_ for p_ in model.parameters() if p_.requires_grad], 1.0
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        visits.setdefault(i, []).append(out.loss.item())
        print(f"[train] {step+1:3d} | s{i} | {out.loss.item():.4f} | {args.lr:.1e}", flush=True)

    peak = torch.cuda.max_memory_allocated(args.device) / 2**30
    reserved = torch.cuda.max_memory_reserved(args.device) / 2**30
    pairs = [(v[0], v[-1]) for v in visits.values() if len(v) >= 2]
    if pairs:
        improved = sum(1 for a, b in pairs if b < a)
        for a, b in pairs:
            print(f"[trend] repeat-visit loss: {a:.4f} -> {b:.4f} ({'DOWN' if b < a else 'up'})",
                  flush=True)
        ok = improved >= (len(pairs) + 1) // 2
        trend = f"{improved}/{len(pairs)} repeat visits improved"
    else:
        ok = visits[0][-1] < visits[0][0]
        trend = f"first={visits[0][0]:.4f} last={visits[0][-1]:.4f}"
    print(f"[summary] model={args.model} class={type(model.base_model.model).__name__} "
          f"device={args.device} steps={args.steps}", flush=True)
    print(f"[summary] lora={lora_mode} grad_ckpt={gc_status} mask={sorted(mask_modes)}", flush=True)
    print(f"[summary] trend: {trend}", flush=True)
    print(f"[summary] peak_mem_allocated={peak:.2f}GiB reserved={reserved:.2f}GiB", flush=True)
    print(f"SMOKE RESULT: {'OK' if ok else 'MARGINAL'}", flush=True)


if __name__ == "__main__":
    main()
