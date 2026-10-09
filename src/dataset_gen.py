"""APIGen-style multi-turn dataset generation with live MCP tool execution.

Pipeline per dialog: seed (instruction/dialogue/scenario) -> user-sim teacher +
assistant teacher (with live tools) -> grounded trajectory -> raw jsonl (append mode).
Modes: tool_use (default), no_tool (irrelevance), error_recovery.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from src import config
from src.llm_client import chat, parse_json_blob
from src.tool_registry import ToolRegistry

NAME = "dataset_gen"

USER_LANG_INSTR = {
    "en": "Write user turns in English.",
    "de": "Schreibe die Nutzernachrichten auf Deutsch.",
    "pl": "Pisz wypowiedzi użytkownika po polsku.",
    "fr": "Écris les messages de l'utilisateur en français.",
    "es": "Escribe los mensajes del usuario en español.",
}
ASSISTANT_LANG_INSTR = {
    "en": "Always answer the user in English.",
    "de": "Antworte dem Nutzer immer auf Deutsch.",
    "pl": "Zawsze odpowiadaj użytkownikowi po polsku.",
    "fr": "Réponds toujours à l'utilisateur en français.",
    "es": "Responde siempre al usuario en español.",
}

PROVIDER_NUDGE = {
    "tomtom": "When multiple providers offer the same capability, prefer the tomtom-* tools.",
    "here": "When multiple providers offer the same capability, prefer the here_* tools.",
    "keyless": "When possible, prefer keyless tools (osrm_route, nominatim_geocode, open_meteo_forecast).",
    "any": "",
}

_lock = threading.Lock()
_counters: dict[str, int] = {}


def _bump(k: str) -> int:
    with _lock:
        _counters[k] = _counters.get(k, 0) + 1
        return _counters[k]


# ---------------------------------------------------------------- seeds
def _seed_text(row: dict) -> str | None:
    """Normalize a seed row's text into a natural task description.

    JSON-blob seeds with origin+destination become 'route from X to Y';
    weak blobs (bare turn instructions, tool-call records) return None and
    are skipped — they invite no-tool answers that fail the quality gate.
    """
    t = (row.get("text") or "").strip()
    if not t.lstrip().startswith("{"):
        return t
    s = t.lstrip()
    if '""' in s:
        s = s.replace('""', '"')  # repair doubled-quote corruption in the seed file
    try:
        obj, _ = json.JSONDecoder().raw_decode(s)
    except Exception:
        return None  # unparseable blob: unusable as a grounded task
    if not isinstance(obj, dict):
        return None
    obj = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else obj

    def _place(p: Any) -> str | None:
        if isinstance(p, dict):
            a = p.get("address") or p.get("name")
            if a:
                return str(a)
            if "lat" in p and ("lon" in p or "lng" in p):
                return f"{p['lat']},{p.get('lon', p.get('lng'))}"
        if isinstance(p, str):
            return p
        return None

    o, d = _place(obj.get("origin")), _place(obj.get("destination"))
    if o and d:
        return f"route from {o} to {d}"
    return None


def load_seeds(limit: int | None = None) -> list[dict]:
    seeds: list[dict] = []
    with open(config.INSTRUCTIONS_V3, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            row = json.loads(line)
            text = _seed_text(row)
            if not text:
                continue
            seeds.append({
                "kind": "instruction",
                "id": f"instr-{i}",
                "text": text,
                "language": row.get("language", "en"),
                "origin": row.get("origin"),
                "destination": row.get("destination"),
                "constraints": row.get("constraints") or [],
                "grounding": row.get("grounding"),
            })
    dialogues = []
    with open(config.DIALOGUES, encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            dialogues.append({
                "kind": "dialogue",
                "id": f"dlg-{row.get('dialogue_id', i)}",
                "text": row.get("first_driver_turn") or next(
                    (t["text"] for t in row.get("turns", []) if t.get("speaker") == "driver"), ""),
                "language": row.get("language", "en"),
                "turn_texts": [t["text"] for t in row.get("turns", []) if t.get("speaker") == "driver"][:6],
            })
    scenarios = []
    with open(config.SCENARIOS, encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            scenarios.append({
                "kind": "scenario",
                "id": f"scen-{i}",
                "card": json.dumps(row, ensure_ascii=False)[:1200],
                "language": row.get("language", "en"),
            })
    return seeds + dialogues + scenarios


EXTRA_TASK_HINTS = [
    "Ask also about parking options at the destination.",
    "Also ask whether it will rain at the destination tomorrow.",
    "Also ask for charging stations along the way (EV).",
    "Also ask about current traffic or incidents on that route.",
    "Also ask how long the route takes and the distance.",
    "Also ask for a route that avoids tolls/motorways.",
    "Also ask for an alternative stopover city roughly halfway.",
    "Also ask about a good restaurant near the destination.",
    "Also ask what the driver can reach within 30 minutes of the destination.",
    "Also ask to compare the route with a different route provider.",
]

NO_TOOL_TOPICS = [
    "general small talk during the drive (music, weather chat without forecast request)",
    "a question about the assistant itself (its name, capabilities)",
    "a completely unrelated task (shopping list, calendar reminder, translation of a sentence)",
    "an opinion question ('do you prefer motorways or country roads?')",
    "a math or unit conversion question ('how many liters is a full tank in gallons?')",
]

ERR_HINTS = [
    "Use a misspelled or nonexistent place name in the first message.",
    "Give an ambiguous destination (just 'Centrum' with no city).",
    "First ask about a place that does not exist, then correct yourself.",
    "Provide coordinates in the wrong format at first (e.g. comma-swapped).",
]


# ---------------------------------------------------------------- generation
_CORE_TOOL_PATTERNS = [
    ("geocode", ("here_geocode", "nominatim_geocode", "tomtom-geocode")),
    ("route", ("here_directions", "osrm_route", "tomtom-routing", "tomtom-waypoint-routing")),
    ("poi", ("here_search_places", "tomtom-poi-search", "tomtom-nearby", "tomtom-fuzzy-search")),
    ("traffic", ("here_traffic_incidents", "tomtom-traffic")),
    ("weather", ("open_meteo_forecast",)),
]


def pick_tools(reg: ToolRegistry, rng: random.Random, preferred: str, n: int = 14) -> list[dict]:
    """Sample n tools, but always include one per core capability (geocode/route/poi/
    traffic/weather) so the extra user requests (parking, weather, traffic, stops) can
    actually be satisfied with the provided catalog instead of dropped or invented."""
    by_name = {s["function"]["name"]: s for s in reg.schemas}
    pref_startswith = {
        "tomtom": ("tomtom-",),
        "here": ("here_",),
        "keyless": ("osrm_", "nominatim_", "open_meteo_"),
    }.get(preferred, ())
    chosen: list[dict] = []
    for _, pats in _CORE_TOOL_PATTERNS:
        cands = [by_name[p] for p in pats if p in by_name]
        if preferred != "any" and pref_startswith:
            pref_c = [s for s in cands if s["function"]["name"].startswith(pref_startswith)]
            if pref_c:
                cands = pref_c
        if cands:
            chosen.append(rng.choice(cands))
    rest = [s for s in reg.schemas if s not in chosen]
    chosen += rng.sample(rest, min(len(rest), max(0, n - len(chosen))))
    return sorted(chosen, key=lambda s: s["function"]["name"])


_ROLE_PREFIX_RE = re.compile(
    r"^\s{0,3}(?:user|driver|passenger|assistant|kierowca|u[żz]ytkownik|pasazer|nutzer|fahrer|"
    r"passagier|conducteur|conductrice|passager|chauffeur|usuario|conductor|pasajero)\s*[:\-–—>]+\s*",
    re.IGNORECASE,
)
_QUOTES = "\"'“”„«»‘’"

# Unambiguous per-language phrases (multi-word, low cross-language overlap) used
# only to REJECT dialogs whose user turns are in the wrong language.
_LANG_HINTS = {
    "fr": re.compile(r"\b(je (souhaite|voudrais|veux|pars|dois)|bonjour,? je|pouvez-vous|"
                     r"itin[ée]raire|j'ai besoin d')", re.IGNORECASE),
    "de": re.compile(r"\b(ich (m[öo]chte|brauche|muss|h[aä]tte)|guten (tag|morgen)|wie komme ich|"
                     r"kannst du mich)", re.IGNORECASE),
    "pl": re.compile(r"\b(proszę o|chciał(by|m)ym|dzień dobry|jak dojadę|poprosz[ęę]|szukam trasy)",
                     re.IGNORECASE),
    "es": re.compile(r"\b(quiero (ir|que|saber)|podr[ií]a usted|buenos d[ií]as|hola,? quer[ií]a|"
                     r"necesito ir)", re.IGNORECASE),
    "en": re.compile(r"\b(could you|can you (tell|help|find)|i (want|need) to (get|go|drive)|"
                     r"good (morning|afternoon),? (i|can|could))", re.IGNORECASE),
}

# API errors can embed the full request URL (incl. apikey=...); never store secrets.
_SECRET_RE = re.compile(
    r"\b(api[-_]?key|access[-_]?token|subscription[-_]?key|token|key)=([A-Za-z0-9_\-.]{6,})",
    re.IGNORECASE,
)


def _redact(text: str) -> str:
    return _SECRET_RE.sub(lambda m: f"{m.group(1)}=***", text)


def _clean_user_msg(text: str) -> str:
    """Strip user-sim artifacts: code fences, role prefixes, wrapping quotes."""
    t = (text or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    for _ in range(3):
        t2 = _ROLE_PREFIX_RE.sub("", t.strip())
        if t2 == t:
            break
        t = t2
    while len(t) >= 2 and t[0] in _QUOTES and t[-1] in _QUOTES:
        t = t[1:-1].strip()
    return t.strip()


def _sys_assistant(lang: str, preferred: str, mode: str) -> str:
    base = (
        "You are a helpful in-car navigation and travel assistant with access to geo tools "
        "(geocoding, routing, POI search, traffic, weather). "
        "Call tools whenever they help answer precisely; batch independent calls in parallel "
        "when useful. After tool results arrive, answer concisely. "
        f"{ASSISTANT_LANG_INSTR[lang]}\n"
        "FINAL-ANSWER DISCIPLINE (mandatory):\n"
        "- Ground EVERY fact in the tool results you received: never state road numbers, highway "
        "names, cities or regions passed through, distances, durations or traffic conditions that "
        "the tool output does not literally contain.\n"
        "- Address EVERY explicit user constraint (avoid tolls/motorways, scenic route, stops, "
        "timing, EV charging, ...). If a constraint was not honored or could not be verified with "
        "a tool, say so plainly instead of claiming it was.\n"
        "- Before promising anything, check which tools are actually provided in this "
        "conversation: if none of them can satisfy a user request (e.g. no parking/POI/traffic "
        "tool is offered), state that limitation plainly and answer what you can with the tools "
        "you do have — never fake a capability or invent the missing data.\n"
        "- If a tool call failed or returned nothing useful, tell the user briefly and naturally "
        "what you could not find out and, if possible, ask for what you need to retry; NEVER fill "
        "the gap from memory or guesswork.\n"
        "- Speak in natural user language: never paste raw JSON, error strings or tool-call "
        "markup into your reply. "
        "Never invent tool results or coordinates: always obtain them from tools."
    )
    if mode in ("tool_use", "error_recovery"):
        base += (" These dialogs are LIVE-DATA tasks about places, routes, traffic or weather: "
                 "you MUST make function calls to the provided tools to ground your answer — "
                 "answering from memory is forbidden, even if you are sure.\n"
                 "TOOL-FLOW DISCIPLINE: read each tool's parameter schema and send complete, "
                 "correctly named arguments in one clean call (a missing or misnamed parameter is "
                 "the most common failure). If a call fails, do not repeat it unchanged: fix the "
                 "arguments or switch to a different tool and retry at most once. If it still "
                 "fails, tell the user plainly what you could not retrieve and answer from the "
                 "results you already have.")
    if mode == "error_recovery":
        base += ("\nRECOVERY GOAL: after the user's slip (misspelled place, ambiguous destination, "
                 "wrong coordinate format) recover FOR REAL: fix the root cause with corrected "
                 "arguments (or one precise clarifying question), make the corrected call succeed, "
                 "and build your final answer on the recovered result. A dialog that ends in "
                 "another failure, or in a question that never gets its route, is a failed dialog.")
    if preferred != "any":
        base += " " + PROVIDER_NUDGE[preferred]
    if mode == "no_tool":
        base += (" IMPORTANT: only call tools when the request genuinely needs live geo data; "
                 "for chitchat/opinion/general-knowledge requests answer directly without tools.")
    return base


def _sim_user(teacher: str, lang: str, seed: dict, mode: str, history: list[dict],
              extra_hint: str, turn_idx: int, max_user_turns: int) -> str:
    if seed["kind"] == "scenario":
        task = "Scenario card:\n" + seed["card"]
    else:
        task = seed["text"] or "(no seed text)"
    hist_txt = "\n".join(f"{m['role']}: {(m.get('content') or '')[:300]}" for m in history[-8:])
    if mode == "no_tool":
        guide = f"Topic for casual conversation (NO navigation request): {extra_hint}"
        stop = "This is pure chitchat; 2-3 user turns max."
    elif mode == "error_recovery":
        guide = f"Navigation task: {task}\nQuirk to introduce: {extra_hint}"
        stop = f"Introduce the quirk in turn {min(2, max_user_turns)}; after the assistant recovers, finish in at most {max_user_turns} user turns."
    else:
        guide = f"Navigation task: {task}\nAdditional request to weave in later: {extra_hint}"
        stop = f"Keep it natural; at most {max_user_turns} user turns total."
    prompt = (
        "You simulate a car driver talking to a navigation assistant. "
        "You are the DRIVER: write in first person, ask/request/correct — NEVER speak as the "
        f"assistant, never offer help or summarize a plan yourself. {USER_LANG_INSTR[lang]}\n"
        f"{guide}\n{stop}\n"
        f"Conversation so far:\n{hist_txt if hist_txt else '(start)'}\n\n"
        "Output ONLY the next user message (no quotes, no role prefix). "
        "If the conversation is finished (assistant fully answered everything), output exactly: DONE"
    )
    if turn_idx == 0:
        prompt += ("\n\nThis is the VERY FIRST message of the conversation: open by stating the "
                   "task in your own words, including the places and constraints from the task. "
                   "NEVER output DONE now.")
    msg = chat(teacher, [{"role": "user", "content": prompt}], temperature=0.9, max_tokens=300)
    return (msg.get("content") or "").strip()


def generate_dialog(reg: ToolRegistry, seed: dict, teacher: str, user_teacher: str | None,
                    lang: str, preferred: str, mode: str, max_turns: int = 8,
                    max_exec: int = 6, seed_rng: random.Random | None = None) -> dict:
    rng = seed_rng or random.Random()
    tools = pick_tools(reg, rng, preferred)
    extra_hint = rng.choice(EXTRA_TASK_HINTS if mode != "no_tool" else NO_TOOL_TOPICS)
    if mode == "error_recovery":
        extra_hint = rng.choice(ERR_HINTS)

    messages: list[dict] = [{"role": "system", "content": _sys_assistant(lang, preferred, mode)}]
    exec_log: list[dict] = []
    max_user_turns = rng.randint(2, 4) if mode != "no_tool" else rng.randint(1, 2)
    turn_idx, executions = 0, 0

    while turn_idx < max(max_turns, max_user_turns * 2 + 2) and turn_idx < 10:
        user_msg = ""
        for sim_try in range(3):
            raw = _sim_user(user_teacher or teacher, lang, seed, mode, messages,
                            extra_hint, turn_idx // 2, max_user_turns)
            user_msg = _clean_user_msg(raw)
            if user_msg and not user_msg.rstrip(".!… ").upper().startswith("DONE"):
                break
            if turn_idx > 0:
                user_msg = ""  # conversation genuinely finished
                break
            user_msg = ""  # turn 0 must produce an opening message: retry, then give up
        if not user_msg:
            break
        messages.append({"role": "user", "content": user_msg})
        turn_idx += 1

        # assistant turn(s) with tool loop
        for _hop in range(3):
            asst = chat(teacher, messages, tools=tools, temperature=0.4, max_tokens=1500)
            tool_calls = asst.get("tool_calls") or []
            if not tool_calls:
                if asst.get("content"):
                    messages.append({"role": "assistant", "content": asst["content"]})
                break
            messages.append({
                "role": "assistant",
                "content": asst.get("content") or "",
                "tool_calls": [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["function"]["name"],
                                  "arguments": tc["function"]["arguments"]}}
                    for tc in tool_calls
                ],
            })
            for tc in tool_calls:
                fn = tc["function"]
                try:
                    args = json.loads(fn["arguments"] or "{}")
                except Exception:
                    args = {}
                if executions >= max_exec:
                    out = {"ok": False, "error": "tool budget exceeded"}
                else:
                    out = reg.execute(fn["name"], args)
                    executions += 1
                exec_log.append({"name": fn["name"], "args": args,
                                 "ok": out["ok"], "cached": out.get("cached", False),
                                 **({"error": _redact(out["error"])} if not out["ok"] else {})})
                payload = _redact(
                    json.dumps(out["result"], ensure_ascii=False) if out["ok"]
                    else json.dumps({"error": out["error"]}, ensure_ascii=False)
                )[:3500]
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": payload})
            if executions >= max_exec:
                break  # budget gone: stop offering tool rounds, close with prose
        else:
            continue
        break

    # guarantee a closing assistant answer (loop may end on a tool message);
    # skip for empty dialogs (0 user turns) — nothing to answer, gate rejects them
    n_users = sum(1 for m in messages if m["role"] == "user")
    if n_users and (messages[-1]["role"] != "assistant"
                    or not (messages[-1].get("content") or "").strip()):
        try:
            final = chat(user_teacher or teacher, messages, temperature=0.4, max_tokens=900)
            if (final.get("content") or "").strip():
                messages.append({"role": "assistant", "content": final["content"]})
        except Exception:  # noqa: BLE001
            pass

    # trim leading system for storage neutrality (kept separately)
    return {
        "id": f"{mode}-{uuid.uuid4().hex[:12]}",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": mode, "lang": lang, "seed_id": seed["id"], "seed_kind": seed["kind"],
        "teacher": teacher, "user_teacher": user_teacher or teacher,
        "preferred_provider": preferred,
        "tools": tools,
        "messages": messages,
        "exec_log": exec_log,
        "stats": {"user_turns": turn_idx, "executions": executions,
                  "ok_exec": sum(1 for e in exec_log if e["ok"]),
                  "n_tool_calls": sum(len(m.get("tool_calls") or []) for m in messages)},
    }


def acceptable(traj: dict) -> tuple[bool, str]:
    """Quality gate before writing a trajectory to disk."""
    msgs = traj["messages"]
    user_turns = sum(1 for m in msgs if m["role"] == "user")
    n_tc = sum(1 for m in msgs if m.get("tool_calls"))
    if user_turns == 0:
        return False, "no user turns (user-sim DONE/empty immediately)"
    lang = traj["lang"]
    for m in [x for x in msgs if x["role"] == "user"][:2]:
        for other, pat in _LANG_HINTS.items():
            if other != lang and pat.search(m.get("content") or ""):
                return False, f"user turn language mismatch: labeled {lang}, looks {other}"
    last = msgs[-1]
    if last["role"] != "assistant" or not (last.get("content") or "").strip():
        return False, "missing final assistant answer"
    if re.search(r'<(?:function|tool)_call\b|"(?:arguments|tool_calls)"\s*:|\btool_call\b',
                 last.get("content") or ""):
        return False, "final answer contains raw tool-call markup"
    if any("budget exceeded" in str(e.get("error") or "") for e in traj["exec_log"]):
        return False, "hit tool budget cap (final answer likely ungrounded)"
    if traj["mode"] == "no_tool":
        if n_tc:
            return False, f"no_tool dialog contains {n_tc} tool_call messages"
        return True, ""
    if traj["mode"] == "tool_use":
        if n_tc == 0:
            return False, "tool_use dialog without any tool call"
        if traj["stats"]["ok_exec"] == 0:
            return False, "tool_use dialog without any successful tool execution"
        if traj["stats"]["executions"] - traj["stats"]["ok_exec"] > 4:
            return False, "teacher thrashing: too many failed tool executions"
        fails = [(e.get("name"), json.dumps(e.get("args") or {}, sort_keys=True))
                 for e in traj["exec_log"] if not e["ok"]]
        if len(fails) != len(set(fails)):
            return False, "identical failed tool call repeated"
        return True, ""
    # error_recovery: a failed tool call, or a clarifying exchange (>=2 user turns)
    n_err = sum(1 for e in traj["exec_log"] if not e["ok"])
    if n_err == 0 and user_turns < 2:
        return False, "error_recovery without error or clarification turn"
    if n_err and not any(e["ok"] for e in traj["exec_log"][next(
            i for i, e in enumerate(traj["exec_log"]) if not e["ok"]):]):
        return False, "error_recovery never executed a successful call after the failure"
    return True, ""


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=str(config.RAW_DIR / "part-0001.jsonl"))
    ap.add_argument("--modes", default="tool_use:0.8,no_tool:0.1,error_recovery:0.1")
    ap.add_argument("--teachers", default=",".join(config.OPENAI_TEACHERS + config.OPENROUTER_TEACHERS))
    ap.add_argument("--user-teachers", default="")  # default: same as teacher
    ap.add_argument("--seed-limit", type=int, default=0, help="limit instruction seeds (0=all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=8)
    args = ap.parse_args()

    weights = {}
    for part in args.modes.split(","):
        k, v = part.split(":")
        weights[k] = float(v)
    teachers = [t for t in args.teachers.split(",") if t]
    user_teachers = [t for t in args.user_teachers.split(",") if t] or [None]

    seeds = load_seeds(args.seed_limit or None)
    reg = ToolRegistry()
    rng = random.Random(42)
    done = 0

    def job(i: int) -> dict:
        why = ""
        for attempt in range(3):
            mode = rng.choices(list(weights), weights=list(weights.values()))[0]
            seed = rng.choice(seeds)
            lang = rng.choice(config.LANGS)
            # non-English seeds keep their own language: simulating a French dialogue
            # in Spanish produces mixed-language turns that fail the quality gate
            if seed.get("language") in config.LANGS and seed["language"] != "en":
                lang = seed["language"]
            preferred = rng.choice(["any", "any", "tomtom", "here", "keyless"])
            teacher = rng.choice(teachers)
            traj = generate_dialog(reg, seed, teacher, rng.choice(user_teachers), lang,
                                   preferred, mode, max_turns=args.max_turns,
                                   seed_rng=random.Random(1000 + i * 10 + attempt))
            ok, why = acceptable(traj)
            if ok:
                return traj
            with _lock:
                print(f"[{NAME}] job {i} attempt {attempt + 1} rejected: {why} "
                      f"(mode={mode} teacher={teacher})", file=sys.stderr, flush=True)
        raise RuntimeError(f"unacceptable after 3 attempts: {why}")

    try:
        with open(args.out, "a", encoding="utf-8") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(job, i): i for i in range(args.n)}
            for fut in as_completed(futs):
                try:
                    traj = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"[{NAME}] job {futs[fut]} failed: {e}", file=sys.stderr)
                    _bump("gen_fail")
                    continue
                fout.write(json.dumps(traj, ensure_ascii=False) + "\n")
                fout.flush()
                done = _bump("ok")
                print(f"[{NAME}] {done}/{args.n} mode={traj['mode']} lang={traj['lang']} "
                      f"tc={traj['stats']['n_tool_calls']} "
                      f"exec={traj['stats']['ok_exec']}/{traj['stats']['executions']} "
                      f"teacher={traj['teacher']}", flush=True)
    finally:
        reg.close()

    print(f"[{NAME}] done: ok={_counters.get('ok', 0)} failed={_counters.get('gen_fail', 0)} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
