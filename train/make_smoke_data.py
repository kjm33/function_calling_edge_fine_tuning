"""Generate tiny synthetic NEUTRAL-format tool-call jsonl for training smokes.

Rows mirror data/final/*.jsonl schema: {"dialog_id","mode","language","tools",
"messages"} — neutral OpenAI style, tool_calls arguments as JSON strings
(renderers convert to dict-args internally where a template requires it).

Defaults: train/smoke_data.jsonl (16 rows: tool_call, one multi-tool-call turn,
no-tool turns) + train/smoke_val.jsonl (4 rows). --dup N replicates the train
split (distinct dialog_ids) for step-count smokes.

Usage:
    train/venv/bin/python train/make_smoke_data.py
    train/venv/bin/python train/make_smoke_data.py --dup 4 \
        --out-train train/smoke_train64.jsonl
"""

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "geocode",
            "description": "Resolve a free-form address or place name to coordinates.",
            "parameters": {
                "type": "object",
                "properties": {"address": {"type": "string", "description": "Address or place name"}},
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
    {
        "type": "function",
        "function": {
            "name": "search_poi",
            "description": "Search points of interest near a coordinate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "lat": {"type": "number"},
                    "lng": {"type": "number"},
                },
                "required": ["query", "lat", "lng"],
            },
        },
    },
]

SYSTEM = (
    "You are a navigation assistant with access to map tools. "
    "Use the provided tools when they help answer the user."
)

J = lambda obj: json.dumps(obj, ensure_ascii=False)


def tool_call_row(dialog_id, mode, language, place, addr, lat, lng, second_call=None):
    calls = [{"id": "call_1", "type": "function",
              "function": {"name": "geocode", "arguments": J({"address": addr})}}]
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Where is {place}?" if language == "en" else
            (f"Wo liegt {place}?" if language == "de" else f"Gdzie jest {place}?")},
        {"role": "assistant", "content": "", "tool_calls": calls},
    ]
    if second_call:
        origin = second_call["origin"]
        msgs[2]["tool_calls"].append({
            "id": "call_2", "type": "function",
            "function": {"name": "get_route", "arguments": J({
                "origin": {"lat": origin[0], "lng": origin[1]},
                "destination": {"lat": lat, "lng": lng},
                "avoid_tolls": second_call.get("avoid_tolls", False)})}},
        )
        msgs.append({"role": "tool", "tool_call_id": "call_2",
                     "content": J({"distance_km": second_call["dist"], "duration_min": second_call["dur"]})})
    msgs.append({"role": "tool", "tool_call_id": "call_1",
                 "content": J({"items": [{"title": place, "lat": lat, "lng": lng}]})})
    final = f"{place} is at latitude {lat}, longitude {lng}."
    if second_call:
        final += (f" The drive from the origin is {second_call['dist']} km "
                  f"and takes about {second_call['dur']} minutes.")
    msgs.append({"role": "assistant", "content": final})
    return {"dialog_id": dialog_id, "mode": mode, "language": language,
            "tools": TOOLS, "messages": msgs}


def no_tool_row(dialog_id, language, place, lat, lng):
    q = {"en": f"Where is {place}?", "de": f"Wo liegt {place}?", "pl": f"Gdzie jest {place}?"}[language]
    a = {"en": f"{place} is at latitude {lat}, longitude {lng}.",
         "de": f"{place} liegt bei Breitengrad {lat} und Längengrad {lng}.",
         "pl": f"{place} znajduje się na szerokości {lat} i długości {lng}."}[language]
    return {"dialog_id": dialog_id, "mode": "no_tool", "language": language,
            "tools": TOOLS,  # tools available but not needed — train restraint
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]}


def no_tools_field_row(dialog_id, place, lat, lng):
    return {"dialog_id": dialog_id, "mode": "no_tool", "language": "en",
            "tools": [],
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
            ]}


def poi_row(dialog_id, place, query, lat, lng, n):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Find {query} near {place}."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "search_poi", "arguments": J({"query": query, "lat": lat, "lng": lng})}}]},
        {"role": "tool", "tool_call_id": "call_1",
         "content": J({"items": [{"name": f"{query} {i + 1}", "distance_m": 200 * (i + 1)} for i in range(n)]})},
        {"role": "assistant", "content":
            f"I found {n} {query} options near {place}; the closest is {query} 1, about 200 m away."},
    ]
    return {"dialog_id": dialog_id, "mode": "tool_call", "language": "en",
            "tools": TOOLS, "messages": msgs}


def train_rows():
    rows = [
        tool_call_row("smoke-t01", "multi_tool_call", "en", "Brandenburger Tor", "Brandenburger Tor, Berlin",
                      52.5163, 13.3777, second_call={"origin": (52.52, 13.405), "dist": 2.4, "dur": 9}),
        tool_call_row("smoke-t02", "tool_call", "en", "Rynek Glowny", "Krakow Main Square", 50.0616, 19.9373),
        tool_call_row("smoke-t03", "tool_call", "en", "Eiffel Tower", "Eiffel Tower, Paris", 48.8584, 2.2945),
        tool_call_row("smoke-t04", "tool_call", "en", "Sagrada Familia", "Sagrada Familia, Barcelona", 41.4036, 2.1744),
        tool_call_row("smoke-t05", "tool_call", "de", "Rijksmuseum", "Rijksmuseum, Amsterdam", 52.3600, 4.8852),
        tool_call_row("smoke-t06", "tool_call", "pl", "Charles Bridge", "Charles Bridge, Prague", 50.0865, 14.4114),
        poi_row("smoke-t07", "Golden Gate Bridge", "coffee", 37.8199, -122.4783, 3),
        poi_row("smoke-t08", "Tokyo Tower", "pharmacy", 35.6586, 139.7454, 2),
        no_tool_row("smoke-t09", "en", "Big Ben", 51.5007, -0.1246),
        no_tool_row("smoke-t10", "de", "Brandenburger Tor", 52.5163, 13.3777),
        no_tool_row("smoke-t11", "pl", "Wawel", 50.0540, 19.9354),
        no_tools_field_row("smoke-t12", "Paris", 48.8566, 2.3522),
        tool_call_row("smoke-t13", "tool_call", "en", "Louvre", "Louvre Museum, Paris", 48.8606, 2.3376),
        tool_call_row("smoke-t14", "tool_call", "en", "Colosseum", "Colosseum, Rome", 41.8902, 12.4922),
        poi_row("smoke-t15", "Prague Castle", "ATM", 50.0903, 14.4032, 4),
        no_tool_row("smoke-t16", "en", "Statue of Liberty", 40.6892, -74.0445),
    ]
    return rows


def val_rows():
    return [
        tool_call_row("smoke-v01", "tool_call", "en", "Acropolis", "Acropolis, Athens", 37.9715, 23.7267),
        tool_call_row("smoke-v02", "tool_call", "de", "Schloss Neuschwanstein", "Neuschwanstein Castle", 47.5576, 10.7498),
        no_tool_row("smoke-v03", "en", "Christ the Redeemer", -22.9519, -43.2105),
        poi_row("smoke-v04", "Vatican", "restaurant", 41.9029, 12.4534, 5),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-train", default="train/smoke_data.jsonl")
    p.add_argument("--out-val", default="train/smoke_val.jsonl")
    p.add_argument("--dup", type=int, default=1,
                   help="replicate train rows N times (suffix in dialog_id) for longer smokes")
    a = p.parse_args()

    for path, rows in ((a.out_train, train_rows()), (a.out_val, val_rows())):
        fp = ROOT / path
        fp.parent.mkdir(parents=True, exist_ok=True)
        with fp.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[write] {fp} rows={len(rows)}")

    if a.dup > 1:
        fp = ROOT / a.out_train
        out = fp.with_name(fp.stem + f"x{a.dup}{fp.suffix}")
        base = train_rows()
        with out.open("w") as f:
            for d in range(a.dup):
                for r in base:
                    r = dict(r, dialog_id=f"{r['dialog_id']}-d{d}")
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[write] {out} rows={len(base) * a.dup}")


if __name__ == "__main__":
    main()
