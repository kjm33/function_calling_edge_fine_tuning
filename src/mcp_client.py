"""Minimal MCP stdio JSON-RPC 2.0 client (spawn server, initialize, list/call tools)."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from typing import Any


class MCPError(RuntimeError):
    pass


class StdioMCPClient:
    def __init__(self, server_name: str, cmd: list[str], env: dict[str, str] | None = None,
                 timeout_s: float = 120.0):
        self.server_name = server_name
        self.cmd = cmd
        self.env = env or {}
        self.timeout_s = timeout_s
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._id = 0
        self.tools: list[dict[str, Any]] = []

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "StdioMCPClient":
        full_env = os.environ.copy()
        full_env.update(self.env)
        self._proc = subprocess.Popen(
            self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=full_env, cwd=os.getcwd(),
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "fc-finetune-pipeline", "version": "0.1.0"},
        })
        self._notify("notifications/initialized")
        self.refresh_tools()
        return self

    def _drain_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        for _ in iter(self._proc.stderr.readline, b""):
            pass  # keep server stderr from blocking; could log if needed

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def _ensure_alive(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            self.start()

    # -- protocol ----------------------------------------------------------
    def _send(self, payload: dict) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write((json.dumps(payload) + "\n").encode())
        self._proc.stdin.flush()

    def _read(self) -> dict:
        assert self._proc and self._proc.stdout
        line = self._proc.stdout.readline()
        if not line:
            raise MCPError(f"[{self.server_name}] server closed stdout")
        return json.loads(line)

    def _request(self, method: str, params: dict | None = None) -> Any:
        with self._lock:
            self._ensure_alive()
            self._id += 1
            rid = self._id
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
            deadline = time.time() + self.timeout_s
            while time.time() < deadline:
                msg = self._read()
                if msg.get("id") != rid:
                    continue  # notifications/responses from server out of band
                if "error" in msg:
                    raise MCPError(f"[{self.server_name}] {method}: {msg['error']}")
                return msg.get("result")
            raise MCPError(f"[{self.server_name}] timeout waiting for {method}")

    def _notify(self, method: str) -> None:
        self._send({"jsonrpc": "2.0", "method": method})

    # -- tools -------------------------------------------------------------
    def refresh_tools(self) -> list[dict[str, Any]]:
        res = self._request("tools/list", {})
        self.tools = res.get("tools", [])
        return self.tools

    def call_tool(self, name: str, arguments: dict) -> Any:
        res = self._request("tools/call", {"name": name, "arguments": arguments})
        if res.get("isError"):
            raise MCPError(f"[{self.server_name}] tool {name} errored: {res}")
        content = res.get("content", [])
        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
        return "\n".join(texts) if texts else res

    # -- helpers -----------------------------------------------------------
    def tool_schemas(self) -> list[dict]:
        """OpenAI-style function schemas from MCP tool definitions."""
        out = []
        for t in self.tools:
            out.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", "")[:2000],
                    "parameters": t.get("inputSchema", {"type": "object", "properties": {}}),
                },
            })
        return out
