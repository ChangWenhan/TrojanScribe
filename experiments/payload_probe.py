#!/usr/bin/env python
"""Single-target payload-generation probe (pre-flight for generation-budget
incidents).

For one shared HotpotQA target it samples the authority-style payload prompt
several times per setting and reports how many candidates are usable under the
same acceptance rule as payload._gen_i (non-empty, >40 chars). This is the
check for the gpt-oss-20b incident (2026-09-10): its harmony reasoning channel
can consume the whole candidate budget, so the final content comes back empty
and authority/bio candidates silently vanish from the pool.

Usage (serve the model first, from experiments/):
  python payload_probe.py --model gpt-oss-20b
  python payload_probe.py --model gpt-oss-20b --efforts medium,low --budgets 256,768
  python payload_probe.py --model gpt-oss-20b --via-backend \
      --top-level-kwargs '{"reasoning_effort": "low"}' --budgets 768
  python payload_probe.py --model gpt-oss-20b --dry-run

Raw mode calls the OpenAI API directly (top-level reasoning_effort, mirroring
vLLM's harmony path). --via-backend instead exercises the real LLMBackend /
AGENTIC_RAG_TOP_LEVEL_KWARGS code path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import openai

from agentic_rag.agents.poisoned.payload import (
    PayloadGenerator,
    _AUTHORITY_PROMPT,
    payload_max_tokens,
)
from agentic_rag.llm import LLMBackend, LLMConfig

SHARED_TARGETS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "targets", "hotpotqa.json",
)


def load_target(qid: str | None) -> tuple[str, str, str, str]:
    data = json.load(open(SHARED_TARGETS))
    qid = qid or next(iter(data))
    rec = data[qid]
    question = rec["question"]
    wrong = rec["incorrect answer"]
    gold = rec["correct answer"]
    return qid, question, wrong, gold


def usable(text: str) -> bool:
    text = (text or "").strip()
    return bool(text) and len(text) > 40


def sample_raw(client, model, prompt, n, max_tokens, effort):
    texts = []
    for _ in range(n):
        extra_body = {"reasoning_effort": effort} if effort else None
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        texts.append((resp.choices[0].message.content or "").strip())
    return texts


def sample_backend(llm, prompt, n, max_tokens):
    return [llm.complete(prompt, max_tokens=max_tokens, temperature=0.9).strip() for _ in range(n)]


def report(setting, texts):
    ok = [t for t in texts if usable(t)]
    lens = [len(t) for t in ok]
    avg = sum(lens) / len(lens) if lens else 0
    verdict = "PASS" if len(ok) == len(texts) else ("PARTIAL" if ok else "FAIL")
    print(f"  {setting:36} usable={len(ok)}/{len(texts)}  avg_len={avg:5.0f}  {verdict}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="served model name, e.g. gpt-oss-20b")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--qid", default=None)
    ap.add_argument("--n", type=int, default=8, help="samples per setting")
    ap.add_argument("--efforts", default="medium,low",
                    help="reasoning_effort values; 'default' omits the field")
    ap.add_argument("--budgets", default="256,768", help="max_tokens values")
    ap.add_argument("--via-backend", action="store_true",
                    help="exercise LLMBackend + AGENTIC_RAG_TOP_LEVEL_KWARGS instead of raw API")
    ap.add_argument("--top-level-kwargs", default='{"reasoning_effort": "low"}',
                    help="env value for AGENTIC_RAG_TOP_LEVEL_KWARGS in --via-backend mode")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    qid, question, wrong, gold = load_target(args.qid)
    prompt = _AUTHORITY_PROMPT.format(question=question, wrong=wrong)
    budgets = [int(b) for b in args.budgets.split(",")]
    efforts = [None if e == "default" else e for e in args.efforts.split(",")]
    is_reasoning = "gpt-oss" in args.model.lower()

    print(f"[probe] model={args.model} qid={qid} n={args.n}")
    print(f"[probe] topic_entity={PayloadGenerator._topic_entity(question)!r}")
    print(f"[probe] wrong={wrong!r}")
    print(f"[probe] payload max_tokens env default = {payload_max_tokens()}")

    if args.dry_run:
        print("[probe] prompt:\n" + prompt)
        print("[probe] settings:",
              [(e or "default", b) for e in efforts for b in budgets])
        return

    if not is_reasoning:
        print("[probe][warn] model is not gpt-oss; reasoning_effort may be ignored/unsupported")

    if args.via_backend:
        os.environ["AGENTIC_RAG_TOP_LEVEL_KWARGS"] = args.top_level_kwargs
        llm = LLMBackend(LLMConfig(base_url=args.base_url, model=args.model))
        print(f"[probe] via-backend top_level={args.top_level_kwargs}")
        for budget in budgets:
            texts = sample_backend(llm, prompt, args.n, budget)
            ok = report(f"backend budget={budget}", texts)
            if ok:
                print("    sample:", ok[0][:160].replace("\n", " "))
            else:
                print("    raw empty content sample:", repr(texts[0][:80]) if texts else "(none)")
        return

    client = openai.OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=300.0)
    for effort in efforts:
        for budget in budgets:
            texts = sample_raw(client, args.model, prompt, args.n, budget, effort)
            ok = report(f"effort={effort or 'default':8} budget={budget:4}", texts)
            if ok:
                print("    sample:", ok[0][:160].replace("\n", " "))
            else:
                print("    raw empty content sample:", repr(texts[0][:80]) if texts else "(none)")


if __name__ == "__main__":
    main()
