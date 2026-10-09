"""Keyless geo tools: OSRM routing, Nominatim geocoding, Open-Meteo weather (no API keys)."""
from __future__ import annotations

import json
from typing import Any

import httpx

_client = httpx.Client(timeout=30, headers={"User-Agent": "fc-finetune-pipeline/0.1 (research)"})

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "osrm_route",
            "description": "Compute a driving route between coordinates using OSRM (OpenStreetMap). Returns distance in km and duration in minutes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coordinates": {"type": "string", "description": "Semicolon-separated 'lng,lat' pairs (lon first!), e.g. '13.405,52.52;11.582,48.135'"},
                },
                "required": ["coordinates"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nominatim_geocode",
            "description": "Geocode an address or place name using OpenStreetMap Nominatim. Keyless.",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Free-text address or place"},
                    "limit": {"type": "integer", "description": "Max results", "default": 3},
                },
                "required": ["q"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_meteo_forecast",
            "description": "Get weather forecast for coordinates using Open-Meteo (keyless). Returns hourly temperature, precipitation, wind for next 24h plus daily summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "Latitude"},
                    "longitude": {"type": "number", "description": "Longitude"},
                    "days": {"type": "integer", "description": "Forecast days (1-7)", "default": 2},
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
]


def call_keyless_tool(name: str, args: dict) -> dict:
    if name == "osrm_route":
        r = _client.get(
            "https://router.project-osrm.org/route/v1/driving/" + args["coordinates"],
            params={"overview": "false"},
        )
        r.raise_for_status()
        route = r.json().get("routes", [{}])[0]
        return {
            "distance_km": round(route.get("distance", 0) / 1000, 1),
            "duration_min": round(route.get("duration", 0) / 60),
        }
    if name == "nominatim_geocode":
        r = _client.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": args["q"], "format": "json", "limit": args.get("limit", 3)},
        )
        r.raise_for_status()
        return {"results": [
            {"name": it.get("display_name"), "lat": it.get("lat"), "lng": it.get("lon")}
            for it in r.json()
        ]}
    if name == "open_meteo_forecast":
        r = _client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": args["latitude"], "longitude": args["longitude"],
                "hourly": "temperature_2m,precipitation,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "forecast_days": min(args.get("days", 2), 7), "timezone": "auto",
            },
        )
        r.raise_for_status()
        d = r.json()
        h = d.get("hourly", {})
        return {
            "timezone": d.get("timezone"),
            "hourly_next12": [
                {"time": h.get("time", [])[i], "temp_c": h.get("temperature_2m", [])[i],
                 "precip_mm": h.get("precipitation", [])[i], "wind_kmh": h.get("wind_speed_10m", [])[i]}
                for i in range(min(12, len(h.get("time", []))))
            ],
            "daily": d.get("daily", {}),
        }
    raise ValueError(f"Unknown keyless tool: {name}")


def call_keyless_tool_str(name: str, args: dict) -> str:
    return json.dumps(call_keyless_tool(name, args), ensure_ascii=False)


if __name__ == "__main__":
    print(call_keyless_tool_str("osrm_route", {"coordinates": "13.405,52.52;11.582,48.135"}))
