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
from agentic_rag.kb.store import BgeEmbeddingFunction  # noqa: E402
from agentic_rag.llm import LLMBackend, load_llm_config  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _EmbedOnlyStore:
    """Just the embedding function: generation does not need the KB.

    PayloadGenerator only touches the store through `embed()` and, as a
    fallback, `search()` (when a target carries no true_paragraph). Opening
    ChromaDB from every generation lane would (a) add a SQLite lock next to the
    evaluation process and (b) make parallel lanes contend for nothing. Passing
    --kb restores the real store for the rare case that needs the lookup.
    """

    def __init__(self, model_path: str, dim: int = 768):
        self._ef = BgeEmbeddingFunction(model_path, dim=dim)

    def embed(self, texts):
        return self._ef.embed(texts)

    def search(self, *a, **k):
        raise RuntimeError(
            "this target has no true_paragraph and generation was started without "
            "the KB; re-run gen_ours_poison.py with --kb")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="attacker/victim backbone")
    ap.add_argument("--dc", default=os.path.join(REPO, "data", "targets",
                                                 "dc_hotpot_100.json"))
    ap.add_argument("--volume", type=int, default=8)
    ap.add_argument("--variant", default="cluster")
    ap.add_argument("--out", default="")
    ap.add_argument("--refill-rounds", type=int, default=6)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("AGENTIC_RAG_GEN_WORKERS", "4")),
                    help="questions generated concurrently; each question still "
                         "draws AGENTIC_RAG_SAMPLE_WORKERS candidates at a time, "
                         "so in-flight requests = workers x sample_workers")
    ap.add_argument("--kb", action="store_true",
                    help="open the real knowledge store (needed only when a target "
                         "has no true_paragraph); default is embeddings only")
    ap.add_argument("--no-gold-leak", action="store_true",
                    help="drop candidate texts that state the correct answer as a "
                         "live fact (marking it obsolete is still allowed). Tested "
                         "on llama-3.1-8b, where such texts act as evidence FOR the "
                         "correct answer.")
    args = ap.parse_args()

    dc = json.load(open(args.dc))
    config = load_config()
    config["llm"]["local"]["model"] = args.model
    if args.kb:
        store = get_store(config)
    else:
        store = _EmbedOnlyStore(config["embedding"]["model_path"],
                                config["embedding"].get("dim", 768))
    llm = LLMBackend(load_llm_config(config, "agent"))
    payload = PayloadGenerator(llm, store, refill_rounds=args.refill_rounds)

    writes, per_q = [], {}
    t0 = time.time()
    print(f"[ours] {len(dc)} questions, volume={args.volume}, variant={args.variant}, "
          f"model={args.model}, gen_workers={args.workers}, "
          f"sample_workers={payload.sample_workers}")

    def one(qid, rec):
        topic = rec.get("topic") or (rec.get("supporting_titles") or ["topic"])[0]
        return qid, payload.generate(
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

    # Questions are independent: generate them concurrently (each one already
    # fans out its candidate sampling), but write the file in the D_c order so
    # the sampler's pick order per question - which the dose arms rely on when
    # they take the first k texts - is preserved exactly.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    out_texts: dict[str, list] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(one, qid, rec): qid for qid, rec in dc.items()}
        for f in as_completed(futs):
            qid, texts = f.result()          # propagates a real error
            out_texts[qid] = texts
            done += 1
            if done % 10 == 0 or done == len(dc):
                print(f"[ours]   {done}/{len(dc)}  ({time.time()-t0:.0f}s)")

    # Shortfalls: the single-style arm (mono) and embed_hybrid can run out of
    # candidates for a question, because the no-gold-leak gate rejects passages
    # that paraphrase the question together with the correct answer. Two graded
    # repairs, both recorded in the meta, so the dose stays exactly --volume:
    #   1) draw much harder for that one question (24 rounds, serial);
    #   2) if still short, generate that question WITHOUT the gate and mark it.
    short = [q for q in dc if len(out_texts.get(q) or []) != args.volume]
    gate_fallback: list[str] = []
    if short:
        print(f"[ours] {len(short)} question(s) short of volume, retrying serially")
        payload.refill_rounds = max(args.refill_rounds, 24)
        for qid in short:
            _, texts = one(qid, dc[qid])
            if len(texts) == args.volume:
                out_texts[qid] = texts
                print(f"[ours]   {qid}: filled after harder draw ({len(texts)})")
        still = [q for q in short if len(out_texts.get(q) or []) != args.volume]
        for qid in still:
            rec = dc[qid]
            topic = rec.get("topic") or (rec.get("supporting_titles") or ["topic"])[0]
            texts = payload.generate(
                question=rec["question"], wrong_answer=rec["wrong"], topic=topic,
                variant=args.variant, n=args.volume, gold_answer=rec["gold"],
                qid=qid, supporting_titles=rec.get("supporting_titles") or [],
                composition="killslot_only",
                true_paragraph=rec.get("true_paragraph") or "",
                no_gold_leak=False,          # documented per-question exception
            )
            if len(texts) == args.volume:
                out_texts[qid] = texts
                gate_fallback.append(qid)
                print(f"[ours]   {qid}: filled only WITHOUT the gold-leak gate "
                      f"(recorded)")
        bad = [q for q in dc if len(out_texts.get(q) or []) != args.volume]
        if bad:
            # A few D_c "wrong answers" are dirty ("The incorrect answer is: ...",
            # unbalanced quotes), so a single-template arm can never make the
            # model reproduce the whole string. The older protocol hit the same
            # wall (mono averaged 3.6/8 chunks) and wrote what it had. Do the
            # same, record every shortfall, and never pretend the dose is 8.
            print(f"[ours] WARNING {len(bad)} question(s) below volume after all "
                  f"retries; writing what was found and recording it "
                  f"({[(q, len(out_texts.get(q) or [])) for q in bad[:3]]} ...)")
            for q in bad:
                out_texts.setdefault(q, [])

    for qid, rec in dc.items():
        texts = out_texts[qid]
        per_q[qid] = len(texts)
        for t in texts:
            writes.append({"qid": qid, "text": t, "target": rec["question"],
                           "variant": args.variant})

    out = args.out or os.path.join(REPO, "results",
                                   f"dc_ours_{args.model}.json")
    blob = {
        "meta": {"baseline": "TrojanScribe (ours)", "variant": args.variant,
                 "model": args.model, "per_target_chunks": args.volume,
                 "n_targets": len(dc), "n_writes": len(writes),
                 "no_gold_leak": bool(args.no_gold_leak),
                 "gate_fallback_qids": gate_fallback,
                 "short_qids": {q: len(out_texts.get(q) or [])
                                for q in dc if len(out_texts.get(q) or []) != args.volume},
                 "mean_dose": round(sum(len(v or []) for v in out_texts.values())
                                    / max(1, len(out_texts)), 2),
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
