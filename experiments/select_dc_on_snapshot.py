"""Re-select D_c on the SNAPSHOT environment (not the pristine KB).

Why this exists
---------------
D_c was selected on the pristine corpus, but every condition runs on
"corpus + snapshot". The snapshot changes retrieval, so questions that were
clean-correct before are not necessarily clean-correct after: the first clean
run on gpt-oss came out 93/100. A protocol whose clean arm is not 100/100 has a
denominator that no longer means what it says.

So: measure clean on each model's OWN snapshot over the whole 206-question
candidate pool, keep the questions every model still answers correctly, and
freeze N of them. Then rewrite D_c and subset the snapshots accordingly.

Writes:
  data/targets/dc_on_snapshot_<model>.json   (per-model clean results on the pool)
  data/targets/dc_<dataset>_<n>.json + .meta (final frozen D_c)
  data/targets/snapshot_<dataset>_<model>.json (rewritten, subset to D_c)

Usage
  cd experiments
  bash serve_model.sh gpt-oss-20b
  AGENTIC_RAG_TOP_LEVEL_KWARGS='{"reasoning_effort":"low"}' \
    python select_dc_on_snapshot.py --model gpt-oss-20b --mode measure
  # repeat per model, then
  python select_dc_on_snapshot.py --mode finalize --n 100
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import get_store, load_config  # noqa: E402

from agentic_rag.agents.base import SYSTEM_VICTIM, victim_system_prompt, ReActAgent, make_kb_search_tool  # noqa: E402
from agentic_rag.eval import unified  # noqa: E402
from agentic_rag.llm import load_llm_config  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = {"hotpot": ("data/chroma", "hotpot_kb"),
            "musique": ("data/chroma_musique", "musique_kb")}
SNAPSHOT_AUTHOR = "system"
POISON_AUTHOR = "doc-consolidator"
SNAP_TAG = {"snapshot_of": POISON_AUTHOR}
MODELS = ["xlam-2-8b", "qwen3-4b-instruct", "gpt-oss-20b", "llama-3.1-8b"]


def measure(dataset: str, model: str, workers: int) -> None:
    from concurrent.futures import ThreadPoolExecutor

    kb_dir, kb_coll = DATASETS[dataset]
    pool_path = os.path.join(REPO, "data", "targets", f"dc_{dataset}_223_all.json")
    snap_path = os.path.join(REPO, "data", "targets", f"snapshot_{dataset}_{model}.json")
    for p in (pool_path, snap_path):
        if not os.path.exists(p):
            raise SystemExit(f"[sds] missing {p}")
    pool = json.load(open(pool_path))
    snap = json.load(open(snap_path))

    config = load_config()
    config["kb"]["persist_dir"] = kb_dir
    config["kb"]["collection"] = kb_coll
    config["llm"]["local"]["model"] = model
    top_k = int(config["kb"].get("top_k", 8))
    store = get_store(config)

    # KB -> corpus + snapshot only
    store.delete_poison()
    store.delete_writer(POISON_AUTHOR)
    res = store._coll.get(where={"snapshot_of": POISON_AUTHOR})
    if res["ids"]:
        store.delete(res["ids"])
    qids_with_snap = {e["qid"] for e in snap["entries"]}
    n = 0
    for e in snap["entries"]:
        for w in e["writes"]:
            store.write(w["text"], author_id=SNAPSHOT_AUTHOR, source=w["source"],
                        is_poison=False, extra=dict(SNAP_TAG, qid=e["qid"]))
            n += 1
    print(f"[sds] {model}: snapshot rows {n}, questions covered "
          f"{len(qids_with_snap)}/{len(pool)}")

    qs = [q for q in pool if q in qids_with_snap]   # only questions with a snapshot
    class _Q:
        def __init__(self, qid, rec):
            self.qid, self.question, self.answer = qid, rec["question"], rec["gold"]
    items = [_Q(q, pool[q]) for q in qs]
    victims = [ReActAgent(name="victim", llm_cfg=load_llm_config(config, "agent"),
                          store=store, system_prompt=victim_system_prompt(),
                          tools=[make_kb_search_tool(store, top_k)], top_k=top_k)
               for _ in range(min(workers, len(items)))]

    def one(iq):
        i, q = iq
        return q, victims[i % len(victims)].ask(q.question)

    out = {}
    with ThreadPoolExecutor(max_workers=len(victims)) as ex:
        for i, (q, r) in enumerate(ex.map(one, enumerate(items)), 1):
            pred = r.get("answer") or ""
            out[q.qid] = {"pred": pred,
                          "correct": bool(unified.correct(pred, q.answer)),
                          "empty": not pred.strip(),
                          "n_tool_calls": r.get("n_tool_calls", 0),
                          "error": r.get("error")}
            if i % 50 == 0 or i == len(items):
                ok = sum(1 for v in out.values() if v["correct"])
                print(f"[sds]   {i}/{len(items)} (correct {ok})")

    n_ok = sum(1 for v in out.values() if v["correct"])
    path = os.path.join(REPO, "data", "targets",
                        f"dc_on_snapshot_{dataset}_{model}.json")
    with open(path, "w") as fh:
        json.dump({"meta": {"dataset": dataset, "model": model, "n": len(out),
                            "n_correct": n_ok, "window_k": top_k,
                            "snapshot_file": os.path.basename(snap_path),
                            "snapshot_sha256": snap["meta"].get("sha256_snapshot"),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
                   "answers": out}, fh, indent=1)
    print(f"[sds] {model}: clean on its own snapshot {n_ok}/{len(out)} -> {path}")
    store.delete_writer(POISON_AUTHOR)
    res = store._coll.get(where={"snapshot_of": POISON_AUTHOR})
    if res["ids"]:
        store.delete(res["ids"])


def finalize(dataset: str, n: int, seed: int, models: list[str]) -> None:
    per = {}
    for m in models:
        p = os.path.join(REPO, "data", "targets",
                         f"dc_on_snapshot_{dataset}_{m}.json")
        if not os.path.exists(p):
            raise SystemExit(f"[sds] missing {p}")
        per[m] = json.load(open(p))["answers"]
    common = set.intersection(*[set(v) for v in per.values()])
    ok = sorted(q for q in common if all(per[m][q]["correct"] for m in models))
    print(f"[sds] common {len(common)}; clean-correct on ALL models' snapshots: {len(ok)}")
    if len(ok) < n:
        raise SystemExit(f"[sds] only {len(ok)} usable, need {n}. Expand the 223 pool.")
    order = list(ok)
    random.Random(seed).shuffle(order)
    chosen = sorted(order[:n])

    pool_path = os.path.join(REPO, "data", "targets", f"dc_{dataset}_223_all.json")
    src = json.load(open(pool_path)) if os.path.exists(pool_path) else {}
    dc = {}
    for qid in chosen:
        r = dict(src.get(qid, {}))
        if not r:
            r = next(json.load(open(os.path.join(
                REPO, "data", "targets", f"dc_on_snapshot_{dataset}_{m}.json")))
                ["answers"][qid] for m in [models[0]])
        dc[qid] = {"question": r.get("question", ""), "gold": r.get("gold", ""),
                   "wrong": "", "supporting_titles": r.get("supporting_titles", []),
                   "true_paragraph": r.get("true_paragraph", ""),
                   "topic": r.get("topic", "")}
    dc_path = os.path.join(REPO, "data", "targets", f"dc_{dataset}_{n}.json")
    with open(dc_path, "w") as fh:
        json.dump(dc, fh, indent=1, ensure_ascii=False)
    proj = {q: {"question": r["question"], "gold": r["gold"]} for q, r in dc.items()}
    sha_q = hashlib.sha256(json.dumps(proj, sort_keys=True).encode()).hexdigest()
    with open(dc_path.replace(".json", ".meta.json"), "w") as fh:
        json.dump({"dataset": dataset, "n": len(dc), "models": models,
                   "window_k": json.load(open(os.path.join(
                       REPO, "data", "targets",
                       f"dc_on_snapshot_{dataset}_{models[0]}.json")))["meta"]["window_k"],
                   "seed": seed, "sha256_questions": sha_q,
                   "criterion": ("clean-correct on EACH model's own snapshot "
                                 "(not on the pristine KB); all four models"),
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=1)
    print(f"[sds] D_c {len(dc)} -> {dc_path} (sha {sha_q[:12]}…)")

    for m in models:
        sp = os.path.join(REPO, "data", "targets", f"snapshot_{dataset}_{m}.json")
        snap = json.load(open(sp))
        keep = [e for e in snap["entries"] if e["qid"] in set(chosen)]
        snap["entries"] = keep
        snap["meta"]["n_questions"] = len(keep)
        snap["meta"]["n_entries"] = sum(e["n_writes"] for e in keep)
        snap["meta"]["dc_file"] = os.path.basename(dc_path)
        snap["meta"]["dc_sha256_questions"] = sha_q
        body = json.dumps([e["writes"] for e in keep], sort_keys=True).encode()
        snap["meta"]["sha256_snapshot"] = hashlib.sha256(body).hexdigest()
        with open(sp, "w") as fh:
            json.dump(snap, fh, indent=1, ensure_ascii=False)
        print(f"[sds] {m}: snapshot subset {len(keep)} questions / "
              f"{snap['meta']['n_entries']} rows")
    print("[sds] next: refill wrong answers for the new D_c with "
          "prepare_dc_targets.py, then re-run clean per model (must be 100/100)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="hotpot")
    ap.add_argument("--mode", choices=["measure", "finalize"], required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.mode == "measure":
        if not args.model:
            raise SystemExit("[sds] --mode measure needs --model")
        measure(args.dataset, args.model, args.workers)
    else:
        finalize(args.dataset, args.n, args.seed, models)


if __name__ == "__main__":
    main()
