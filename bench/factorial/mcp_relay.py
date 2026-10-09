#!/usr/bin/env python3
"""Stdio MCP relay in front of `gbrain serve` that pins the evidence budget and packing unit.

Every tools/call for `search` or `query` gets token_budget = $RELAY_BUDGET and
return_unit = $RELAY_UNIT, overriding whatever the model asked for. All other traffic passes
through unchanged.
"""

import json
import os
import subprocess
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agents"))
from mcp_client import GBRAIN  # noqa: E402

BUDGET = int(os.environ["RELAY_BUDGET"])
UNIT = os.environ["RELAY_UNIT"]


def pump(src, dst) -> None:
    for line in src:
        dst.write(line)
        dst.flush()


def main() -> None:
    child = subprocess.Popen(GBRAIN, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    threading.Thread(target=pump, args=(child.stdout, sys.stdout), daemon=True).start()
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            child.stdin.write(line)
            child.stdin.flush()
            continue
        if msg.get("method") == "tools/call":
            p = msg.setdefault("params", {})
            if p.get("name") in ("search", "query"):
                args = p.setdefault("arguments", {})
                args["token_budget"] = BUDGET
                args["return_unit"] = UNIT
                line = json.dumps(msg) + "\n"
        child.stdin.write(line if line.endswith("\n") else line + "\n")
        child.stdin.flush()
    child.stdin.close()
    child.wait()


if __name__ == "__main__":
    main()
