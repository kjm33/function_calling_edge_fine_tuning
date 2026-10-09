"""HERE REST API tool wrappers (geocode, reverse, discover, routing v8, traffic v7).

Domain note: HERE APIs live under *.hereapi.com (geocode.search.here.com etc. do not resolve).
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from src import config

GEOCODE = "https://geocode.search.hereapi.com/v1/geocode"
REVGEOCODE = "https://revgeocode.search.hereapi.com/v1/revgeocode"
DISCOVER = "https://discover.search.hereapi.com/v1/discover"
ROUTING = "https://router.hereapi.com/v8/routes"
TRAFFIC = "https://data.traffic.hereapi.com/v7/incidents"

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "here_geocode",
            "description": "Geocode a free-text address or place name to coordinates using HERE. Returns best match with lat/lng, formatted address, and country.",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Address or place name, e.g. 'Brandenburger Tor, Berlin'"},
                    "lang": {"type": "string", "description": "ISO-639-1 language for results, e.g. 'en', 'de', 'pl'"},
                    "limit": {"type": "integer", "description": "Max results (1-100)", "default": 3},
                },
                "required": ["q"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "here_reverse_geocode",
            "description": "Reverse-geocode coordinates to the nearest address / place using HERE.",
            "parameters": {
                "type": "object",
                "properties": {
                    "at": {"type": "string", "description": "Position as 'lat,lng', e.g. '52.5163,13.3777'"},
                    "lang": {"type": "string", "description": "ISO-639-1 language for results"},
                },
                "required": ["at"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "here_search_places",
            "description": "Search for places / POIs by name or category around a location using HERE Discover. Example: 'restaurants near Eiffel Tower'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Free-text place query, e.g. 'charging station'"},
                    "at": {"type": "string", "description": "Search center as 'lat,lng'"},
                    "in_country": {"type": "string", "description": "ISO-3 country code filter, e.g. 'DEU'"},
                    "limit": {"type": "integer", "description": "Max results (1-100)", "default": 5},
                },
                "required": ["q", "at"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "here_directions",
            "description": "Compute a driving/walking/truck route between origin and destination using HERE Routing v8. Returns distance (m), duration (s), and departure/arrival times.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transportMode": {"type": "string", "enum": ["car", "truck", "pedestrian", "bicycle", "scooter"], "description": "Transport mode"},
                    "origin": {"type": "string", "description": "Origin as 'lat,lng'"},
                    "destination": {"type": "string", "description": "Destination as 'lat,lng'"},
                    "via": {"type": "array", "items": {"type": "string"}, "description": "Optional waypoints as 'lat,lng' list"},
                    "departureTime": {"type": "string", "description": "ISO datetime; omit for now"},
                    "routingMode": {"type": "string", "enum": ["fast", "short"], "description": "Fastest or shortest route"},
                    "avoid_features": {"type": "array", "items": {"type": "string", "enum": ["tollRoad", "ferry", "motorway", "tunnel"], "description": "Features to avoid"}},
                    "lang": {"type": "string", "description": "Language for route instructions"},
                },
                "required": ["transportMode", "origin", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "here_traffic_incidents",
            "description": "Get active traffic incidents (accidents, congestion, roadworks) in a circular area using HERE Traffic v7.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Circular area, e.g. 'circle:52.52,13.40;r=10000' (r in meters)"},
                },
                "required": ["location"],
            },
        },
    },
]

_client = httpx.Client(timeout=30)


def _compact_route(data: dict) -> dict:
    routes = []
    for r in data.get("routes", [])[:3]:
        sections = []
        for s in r.get("sections", []):
            dep, arr = s.get("departure", {}), s.get("arrival", {})
            summ = s.get("summary", {})
            sections.append({
                "departure_place": dep.get("place", {}).get("location"),
                "arrival_place": arr.get("place", {}).get("location"),
                "length_km": round(summ.get("length", 0) / 1000, 1),
                "duration_min": round(summ.get("duration", 0) / 60),
                "baseDuration_min": round(summ.get("baseDuration", 0) / 60),
            })
        routes.append(sections)
    return {"routes": routes}


def _compact_geocode(data: dict, limit: int = 3) -> dict:
    items = []
    for it in data.get("items", [])[:limit]:
        items.append({
            "title": it.get("title"),
            "lat": it.get("position", {}).get("lat"),
            "lng": it.get("position", {}).get("lng"),
            "address": it.get("address", {}),
        })
    return {"results": items}


def call_here_tool(name: str, args: dict) -> dict:
    key = config.HERE_API_KEY
    if name == "here_geocode":
        p = {"q": args["q"], "apiKey": key, "limit": args.get("limit", 3)}
        if args.get("lang"):
            p["lang"] = args["lang"]
        r = _client.get(GEOCODE, params=p)
        r.raise_for_status()
        return _compact_geocode(r.json())
    if name == "here_reverse_geocode":
        p = {"at": args["at"], "apiKey": key}
        if args.get("lang"):
            p["lang"] = args["lang"]
        r = _client.get(REVGEOCODE, params=p)
        r.raise_for_status()
        return _compact_geocode(r.json(), limit=1)
    if name == "here_search_places":
        p = {"q": args["q"], "at": args["at"], "apiKey": key, "limit": args.get("limit", 5)}
        if args.get("in_country"):
            p["in"] = f"countryCode:{args['in_country']}"
        r = _client.get(DISCOVER, params=p)
        r.raise_for_status()
        items = []
        for it in r.json().get("items", [])[: p["limit"]]:
            items.append({
                "title": it.get("title"),
                "category": (it.get("categories") or [{}])[0].get("name"),
                "lat": it.get("position", {}).get("lat"),
                "lng": it.get("position", {}).get("lng"),
                "distance_m": it.get("distance"),
            })
        return {"results": items}
    if name == "here_directions":
        p = {
            "transportMode": args["transportMode"],
            "origin": args["origin"],
            "destination": args["destination"],
            "apikey": key,
            "return": "summary",
        }
        if args.get("via"):
            p["via"] = ",".join(args["via"]) if isinstance(args["via"], list) else args["via"]
            p["return"] = "summary"
        if args.get("routingMode"):
            p["routingMode"] = args["routingMode"]
        if args.get("lang"):
            p["lang"] = args["lang"]
        if args.get("avoid_features"):
            p["avoid"] = json.dumps({"features": args["avoid_features"]})
        r = _client.get(ROUTING, params=p)
        r.raise_for_status()
        return _compact_route(r.json())
    if name == "here_traffic_incidents":
        p = {"location": args["location"], "apiKey": key}
        r = _client.get(TRAFFIC, params=p)
        r.raise_for_status()
        incs = []
        for it in r.json().get("incidents", [])[:10]:
            loc = it.get("location", {})
            desc = it.get("description", {})
            incs.append({
                "type": it.get("incidentType"),
                "severity": it.get("severity"),
                "start": it.get("startTime"),
                "road": desc.get("roadName") if isinstance(desc, dict) else None,
                "summary": (desc.get("description") or [{}])[0].get("value") if isinstance(desc, dict) and desc.get("description") else None,
            })
        return {"incidents": incs, "count": len(incs)}
    raise ValueError(f"Unknown HERE tool: {name}")


def call_here_tool_str(name: str, args: dict) -> str:
    return json.dumps(call_here_tool(name, args), ensure_ascii=False)


if __name__ == "__main__":
    print(call_here_tool_str("here_geocode", {"q": "Brandenburger Tor Berlin", "limit": 1}))
    print(call_here_tool_str("here_directions", {"transportMode": "car", "origin": "52.5163,13.3777", "destination": "48.1351,11.5820"}))
