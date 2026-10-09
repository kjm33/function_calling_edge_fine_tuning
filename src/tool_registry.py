"""Unified tool registry: TomTom (official MCP server) + HERE (REST wrapper) + keyless tools.

Provides OpenAI-style tool schemas for prompts and a single execute() with a persistent
arg-keyed JSON cache (protects TomTom's ~2.5k/day routing quota).
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import threading
import time
from typing import Any, Callable

from src import config
from mcp.here_tools import TOOLS as HERE_TOOLS, call_here_tool_str
from mcp.keyless_tools import TOOLS as KEYLESS_TOOLS, call_keyless_tool_str
from src.mcp_client import StdioMCPClient

NODE22 = pathlib.Path.home() / "tools" / "node22" / "bin"
# Override with FCFT_TOOL_CACHE to isolate caches between concurrent gen processes
# (the default shared file is last-writer-wins under concurrent writers).
CACHE_PATH = DATA_CACHE = pathlib.Path(
    os.environ.get("FCFT_TOOL_CACHE") or config.DATA_DIR / "tool_response_cache.json")
_CACHE_VERSION = 1

_lock = threading.Lock()
_cache: dict[str, Any] = {}


def _load_cache() -> None:
    global _cache
    if CACHE_PATH.exists():
        try:
            _cache = json.loads(CACHE_PATH.read_text())
        except Exception:
            _cache = {}
    else:
        _cache = {}


def _save_cache() -> None:
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(_cache, ensure_ascii=False))
    tmp.replace(CACHE_PATH)


def _cache_key(provider: str, tool: str, args: dict) -> str:
    stable = json.dumps(args, sort_keys=True, ensure_ascii=False)
    return f"{_CACHE_VERSION}:{provider}:{tool}:{hashlib.sha256(stable.encode()).hexdigest()}"


class ToolRegistry:
    def __init__(self, use_tomtom: bool = True, use_here: bool = True, use_keyless: bool = True):
        self.schemas: list[dict[str, Any]] = []
        self.providers: dict[str, str] = {}  # tool name -> provider
        self._executors: dict[str, Callable[[str, dict], str]] = {}
        self.tomtom_client: StdioMCPClient | None = None
        _load_cache()
        if use_keyless:
            for t in KEYLESS_TOOLS:
                self._register(t, "keyless", lambda n, a: call_keyless_tool_str(n, a))
        if use_here:
            for t in HERE_TOOLS:
                self._register(t, "here", lambda n, a: call_here_tool_str(n, a))
        if use_tomtom:
            self._start_tomtom()

    def _register(self, schema: dict, provider: str, fn: Callable[[str, dict], str]) -> None:
        name = schema["function"]["name"]
        self.schemas.append(schema)
        self.providers[name] = provider
        self._executors[name] = fn

    def _start_tomtom(self) -> None:
        import os
        env_path = str(NODE22)
        os.environ["PATH"] = env_path + os.pathsep + os.environ.get("PATH", "")
        self.tomtom_client = StdioMCPClient(
            "tomtom",
            ["npx", "-y", "@tomtom-org/tomtom-mcp@latest"],
            env={"TOMTOM_API_KEY": config.TOMTOM_API_KEY, "PATH": os.environ["PATH"]},
        ).start()
        for t in self.tomtom_client.tool_schemas():
            name = t["function"]["name"]
            # Skip UI/viz tools not useful for text training data.
            if name in ("tomtom-dynamic-map", "tomtom-data-viz"):
                continue
            self._register(t, "tomtom", self._call_tomtom)
        # force compact responses on routing-family tools
        for s in self.schemas:
            if s["function"]["name"].startswith("tomtom-"):
                props = s["function"]["parameters"].get("properties", {})
                if "response_detail" in props:
                    props["response_detail"]["default"] = "compact"

    def _call_tomtom(self, name: str, args: dict) -> str:
        assert self.tomtom_client
        return self.tomtom_client.call_tool(name, args)

    # -- public API ---------------------------------------------------------
    def execute(self, name: str, args: dict, use_cache: bool = True) -> dict:
        """Execute a tool; returns {ok, result|error, cached}."""
        provider = self.providers.get(name)
        if provider is None:
            return {"ok": False, "error": f"unknown tool {name}", "cached": False}
        args = _normalize_args(name, args)
        key = _cache_key(provider, name, args)
        if use_cache:
            with _lock:
                if key in _cache:
                    return {"ok": True, "result": _cache[key], "cached": True}
        try:
            result = self._executors[name](name, args)
            parsed = json.loads(result) if isinstance(result, str) else result
            compact = _shrink(name, parsed)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"{type(e).__name__}: {e}"[:500], "cached": False}
        if use_cache and compact is not None:
            with _lock:
                _cache[key] = compact
                if len(_cache) % 50 == 0:
                    _save_cache()
        return {"ok": bool(compact is not None), "result": compact, "cached": False}

    def flush_cache(self) -> None:
        with _lock:
            _save_cache()

    def close(self) -> None:
        self.flush_cache()
        if self.tomtom_client:
            self.tomtom_client.stop()

    def get_schema(self, name: str) -> dict | None:
        for s in self.schemas:
            if s["function"]["name"] == name:
                return s
        return None


def _to_latlon_obj(v: Any) -> Any:
    """Coerce any coordinate representation into a {lat, lon} object."""
    if isinstance(v, dict):
        return {"lat": float(v["lat"]), "lon": float(v.get("lon", v.get("lng", v.get("lng", 0))))}
    if isinstance(v, (list, tuple)) and len(v) == 2:
        a, b = float(v[0]), float(v[1])
        return {"lat": a, "lon": b} if a > b else {"lat": b, "lon": a}  # EU coords heuristic
    return v


def _normalize_args(name: str, args: dict) -> dict:
    """Coerce common teacher-argument quirks into schema-valid values."""
    a = dict(args)
    if name == "tomtom-routing":
        for k in ("origin", "destination"):
            if k in a:
                a[k] = _to_latlon_obj(a[k])
        a.pop("locations", None)
    elif name == "tomtom-waypoint-routing":
        if "waypoints" in a and isinstance(a["waypoints"], list):
            a["waypoints"] = [_to_latlon_obj(p) for p in a["waypoints"]]
    elif name == "tomtom-reachable-range":
        for k in ("center", "origin"):
            if k in a:
                a["center"] = _to_latlon_obj(a.pop(k) if k == "origin" else a[k])
                break
    elif name in ("tomtom-nearby", "tomtom-poi-search", "tomtom-fuzzy-search", "tomtom-reverse-geocode"):
        if "lat" in a and "lon" not in a and "lng" in a:
            a["lon"] = a.pop("lng")
        if isinstance(a.get("position"), (list, dict)):
            o = _to_latlon_obj(a.pop("position"))
            a.setdefault("lat", o["lat"]); a.setdefault("lon", o["lon"])
    if name == "osrm_route" and "coordinates" in a and isinstance(a["coordinates"], list):
        a["coordinates"] = ";".join(
            f"{float(p[0]):.6f},{float(p[1]):.6f}" if isinstance(p, (list, tuple))
            else str(p) for p in a["coordinates"]
        )
    # strip null-valued optional args (HERE/TomTom treat null as absent)
    return {k: v for k, v in a.items() if v is not None}


_MAX_LIST = 8
_MAX_STR = 600


def _shrink(name: str, parsed: Any, depth: int = 0) -> Any:
    """Compact large API responses so they fit training context."""
    if parsed is None:
        return None
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except Exception:
            s = parsed.strip()
            return s[:_MAX_STR] + ("…" if len(s) > _MAX_STR else "")
    if isinstance(parsed, list):
        return [_shrink(name, x, depth + 1) for x in parsed[:_MAX_LIST]]
    if isinstance(parsed, dict):
        out = {}
        for k, v in parsed.items():
            if k in ("geometry", "points", "legs", "guidance", "instructions", "addressTags",
                     "dataSources", "entryPoints"):
                continue  # heavy fields useless for FC training
            out[k] = _shrink(name, v, depth + 1)
        return out
    return parsed


if __name__ == "__main__":
    reg = ToolRegistry()
    try:
        print(f"tools={len(reg.schemas)}")
        for s in reg.schemas:
            print(" ", s["function"]["name"], "->", reg.providers[s["function"]["name"]])
    finally:
        reg.close()
