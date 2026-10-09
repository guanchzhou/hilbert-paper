"""Minimal MCP stdio client for gbrain serve, the same server Cursor and Claude Code use."""

import json
import subprocess


class MCP:
    def __init__(self, cmd: list[str]):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, bufsize=1)
        self.n = 0
        self.call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                 "clientInfo": {"name": "bench", "version": "1"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _send(self, msg: dict) -> None:
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()

    def call(self, method: str, params: dict) -> dict:
        self.n += 1
        rid = self.n
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("MCP server closed")
            msg = json.loads(line)
            if msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg["result"]

    def tool(self, name: str, args: dict) -> str:
        res = self.call("tools/call", {"name": name, "arguments": args})
        return "\n".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")

    def close(self) -> None:
        self.p.terminate()


GBRAIN = ["/Users/andreymaltsev/.bun/bin/bun", "/Users/andreymaltsev/.bun/install/global/node_modules/gbrain/src/cli.ts", "serve"]
