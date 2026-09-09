"""Harness-level tool-calling smoke: drives the real LangGraph victim through
make_victim().ask() and fails unless the model actually invoked kb_search.
The OpenAI-level tool_smoke (direct API call) does NOT prove the harness loop
works — Granite-3.1-8B refused tool calls inside create_react_agent until the
SYSTEM_VICTIM wording was hardened (incident 2026-09-08)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import get_store, load_config, make_victim  # noqa: E402


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "xlam-2-8b"
    config = load_config()
    config["llm"]["local"]["model"] = model
    store = get_store(config)
    victim = make_victim(store, config)
    # A TRIVIAL question is the wrong probe: memorization-capable models
    # (qwen3-8b) answer it directly without calling any tool, which is not a
    # harness failure. Use a real long-tail target the model cannot answer
    # from memory (incident 2026-09-08: qwen3-8b mis-flagged TOOL_CALL_FAILED).
    import json as _json
    shared_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "targets",
        "hotpotqa.json")
    rec = _json.load(open(shared_path))
    question = next(iter(rec.values()))["question"]
    r = victim.ask(question)
    calls = r.get("n_tool_calls", 0)
    if calls == 0:
        print("HARNESS_TOOL_MISSING:", repr((r.get("answer") or "")[:200]))
        sys.exit(2)
    print(f"HARNESS_TOOL_OK calls={calls} answer={repr((r.get('answer') or '')[:80])}")


if __name__ == "__main__":
    main()