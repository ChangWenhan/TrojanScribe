"""Generate OUR method's poison texts for D_c (generation only, no evaluation).

Why a dedicated script instead of `longtail_attack.py --phase inject`:
that path runs the full writer agent for every target (3 warm-up rounds plus one
curator task per target), which is exactly the extra benign write-back the new
protocol moves into the shared snapshot. Running it here would add a second
layer of benign rows on top of the snapshot and make our method's KB differ from
the baselines' again.

This script therefore:
  * loads D_c and only calls the payload generator;
  * writes each poison set to a file in the same shape the baseline generators
    use (`variants.cluster.poison_writes`), so `dc_run.py` can replay it;
  * never touches the KB.

Dose is fixed and enforced: exactly `--volume` texts per question. Old runs could
end up with 3.6/8 on average after quality filters; here a shortfall is a hard
error, because "8 per target" has to mean 8.

Usage
  cd experiments
  bash serve_model.sh gpt-oss-20b
  AGENTIC_RAG_TOP_LEVEL_KWARGS='{"reasoning_effort":"low"}' \
    python gen_ours_poison.py --model gpt-oss-20b --volume 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import get_store, load_config  # noqa: E402

from agentic_rag.agents.poisoned.payload import PayloadGenerator  # noqa: E402
from agentic_rag.llm import LLMBackend, load_llm_config  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="attacker/victim backbone")
    ap.add_argument("--dc", default=os.path.join(REPO, "data", "targets",
                                                 "dc_hotpot_100.json"))
    ap.add_argument("--volume", type=int, default=8)
    ap.add_argument("--variant", default="cluster")
    ap.add_argument("--out", default="")
    ap.add_argument("--refill-rounds", type=int, default=6)
    ap.add_argument("--no-gold-leak", action="store_true",
                    help="drop candidate texts that state the correct answer as a "
                         "live fact (marking it obsolete is still allowed). Tested "
                         "on llama-3.1-8b, where such texts act as evidence FOR the "
                         "correct answer.")
    args = ap.parse_args()

    dc = json.load(open(args.dc))
    config = load_config()
    config["llm"]["local"]["model"] = args.model
    store = get_store(config)
    llm = LLMBackend(load_llm_config(config, "agent"))
    payload = PayloadGenerator(llm, store, refill_rounds=args.refill_rounds)

    writes, per_q = [], {}
    t0 = time.time()
    print(f"[ours] {len(dc)} questions, volume={args.volume}, variant={args.variant}, "
          f"model={args.model}")
    for i, (qid, rec) in enumerate(dc.items(), 1):
        topic = rec.get("topic") or (rec.get("supporting_titles") or ["topic"])[0]
        texts = payload.generate(
            question=rec["question"],
            wrong_answer=rec["wrong"],
            topic=topic,
            variant=args.variant,
            n=args.volume,
            gold_answer=rec["gold"],
            qid=qid,
            supporting_titles=rec.get("supporting_titles") or [],
            composition="killslot_only",
            true_paragraph=rec.get("true_paragraph") or "",
            no_gold_leak=args.no_gold_leak,
        )
        if len(texts) != args.volume:
            # a shortfall would silently shrink our dose relative to the "8 per
            # target" the protocol states
            raise SystemExit(
                f"[ours] {qid}: got {len(texts)} texts, expected {args.volume}. "
                f"Raising --refill-rounds or lowering --volume is a protocol "
                f"decision, not something to paper over.")
        per_q[qid] = len(texts)
        for t in texts:
            writes.append({"qid": qid, "text": t, "target": rec["question"],
                           "variant": args.variant})
        if i % 10 == 0 or i == len(dc):
            print(f"[ours]   {i}/{len(dc)}  ({time.time()-t0:.0f}s)")

    out = args.out or os.path.join(REPO, "results",
                                   f"dc_ours_{args.model}.json")
    blob = {
        "meta": {"baseline": "TrojanScribe (ours)", "variant": args.variant,
                 "model": args.model, "per_target_chunks": args.volume,
                 "n_targets": len(dc), "n_writes": len(writes),
                 "no_gold_leak": bool(args.no_gold_leak),
                 "dc_file": os.path.basename(args.dc),
                 "generation_only": True,
                 "note": ("poison texts only: the benign write-back half of the "
                          "attack lives in the shared snapshot"),
                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "targets": {q: {"question": r["question"], "gold": r["gold"],
                        "wrong": r["wrong"]} for q, r in dc.items()},
        "variants": {args.variant: {"poison_writes": writes}},
    }
    with open(out, "w") as fh:
        json.dump(blob, fh, indent=1, ensure_ascii=False)
    print(f"[ours] wrote {len(writes)} texts -> {out}")
    print(f"[ours] per-question minimum: {min(per_q.values())}, "
          f"maximum: {max(per_q.values())}")


if __name__ == "__main__":
    main()
