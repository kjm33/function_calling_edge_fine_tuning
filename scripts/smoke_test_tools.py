"""Smoke test: execute one representative tool per provider through the registry."""
import json
import sys

from src.tool_registry import ToolRegistry

CASES = [
    ("tomtom-geocode", {"query": "Brandenburger Tor Berlin", "limit": 1}),
    ("tomtom-routing", {"origin": {"lat": 52.5163, "lon": 13.3777},
                        "destination": {"lat": 48.1351, "lon": 11.5820},
                        "routeType": "fastest", "travelMode": "car"}),
    ("tomtom-waypoint-routing", {"waypoints": [{"lat": 52.5163, "lon": 13.3777},
                                               {"lat": 50.1109, "lon": 8.6821},
                                               {"lat": 48.1351, "lon": 11.5820}]}),
    ("tomtom-poi-search", {"query": "charging station", "lat": 52.5163, "lon": 13.3777, "limit": 2}),
    ("here_geocode", {"q": "Marienplatz München", "limit": 1}),
    ("here_directions", {"transportMode": "car", "origin": "52.5163,13.3777",
                         "destination": "48.1351,11.5820"}),
    ("osrm_route", {"coordinates": "13.3777,52.5163;11.582,48.1351"}),
    ("open_meteo_forecast", {"latitude": 52.52, "longitude": 13.405, "days": 1}),
]


def main() -> int:
    reg = ToolRegistry()
    fails = 0
    try:
        for name, args in CASES:
            res = reg.execute(name, args)
            status = "OK " if res["ok"] else "ERR"
            body = json.dumps(res.get("result") or res.get("error"), ensure_ascii=False)[:180]
            print(f"[{status}] {name} (cached={res['cached']}): {body}")
            fails += 0 if res["ok"] else 1
    finally:
        reg.close()
    print(f"\n{len(CASES) - fails}/{len(CASES)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
