"""Build the D_c evaluation set: N questions EVERY victim answers correctly with
plain RAG, so all methods share one denominator.

Why
---
Current protocol has per-model denominators (clean-correct 27/31/38/25 out of 60
on HotpotQA; only 16 questions are clean-correct for all four). 22/27 cannot be
compared with 21/38, and the MuSiQue "59" is a frozen long-tail subset, not a
dataset-wide denominator.

D_c fixes that by construction: freeze N questions that every victim answers
correctly BEFORE any attack. Flip stays "was correct, now non-empty wrong"; the
denominator simply stops varying.

Hard rule: D_c is only valid for the exact conditions it was measured under.
The frozen file therefore records
    * the served model name AND the model path the server was started from,
    * the retrieval window k and the chat/request settings,
    * a hash of the KB content it was measured on,
and every downstream run must compare against those. A D_c measured with k=8 is
not valid for a k=4 ablation.

Two stages
----------
  screen : one cheap model over the corpus, over-select a pool.
  verify : all four victims over that pool; keep the ones ALL of them answer
           correctly; freeze exactly N.

The verify stage REQUIRES an externally served model per name: this script does
NOT switch servers (a single vLLM process answers one model). Use
`experiments/serve_model.sh <model>` between runs, or pass --one-model-per-run
and let the driver loop. Refusing otherwise is deliberate: silently sending four
different model names to one server records "one model answered four times" as
"four models agree" (review finding, 2026-10-26).

Usage
  cd experiments
  # stage 1 (server must serve --screen-model)
  bash serve_model.sh gpt-oss-20b
  python build_dc_set.py --dataset hotpot --mode screen --pool 500
  # stage 2: driver serves each model in turn
  python build_dc_set.py --dataset hotpot --mode verify \
      --pool-file ../data/targets/dc_pool_hotpot.json --n 100 --one-model-per-run 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import get_store, load_config  # noqa: E402

from agentic_rag.eval import unified  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = {
    "hotpot": {"corpus": os.path.join(REPO, "data", "hotpot_dev_distractor_v1.json"),
               "kb_dir": "data/chroma", "kb_collection": "hotpot_kb"},
    "musique": {"corpus": os.path.join(REPO, "data", "musique_ans_v1.0_dev.jsonl"),
                "kb_dir": "data/chroma_musique", "kb_collection": "musique_kb"},
}
# The four victims of the D_c protocol. qwen3-8b was retired on 2026-10-26:
# its weights (/mnt/disk/cwh/LLMs/Qwen3-8B) no longer exist on this machine and
# the HF cache is empty. Its successor here is Qwen3-4B-Instruct-2507, the same
# qwen3 architecture the --reasoning-parser qwen3 path already supports, so the
# serving recipe does not change (only the size, 4B instead of 8B).
DEFAULT_MODELS = ["xlam-2-8b", "qwen3-4b-instruct", "gpt-oss-20b", "llama-3.1-8b"]
MODEL_PATHS = {
    "xlam-2-8b": "/mnt/disk/cwh/LLMs/xlam-2-8b-fc-r",
    "qwen3-4b-instruct": "/mnt/disk/cwh/LLMs/Qwen3-4B-Instruct-2507",
    "gpt-oss-20b": "/mnt/disk/cwh/LLMs/gpt-oss-20b",
    "llama-3.1-8b": "/mnt/disk/cwh/LLMs/Meta-Llama-3.1-8B-Instruct",
}
# questions whose gold answer is a bare yes/no are excluded: unified.correct()
# is substring containment, so "No" matches "November" and "Yes" matches
# "yesterday", which would let a wrong-looking answer count as clean-correct.
YESNO = {"yes", "no", "true", "false"}


def served_models(base_url: str) -> list[str]:
    import urllib.request

    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=10) as r:
            return [m["id"] for m in json.loads(r.read()).get("data", [])]
    except Exception:
        return []


def kb_fingerprint(store) -> dict:
    """Content fingerprint of the KB, not just its size.

    A chunk-count comparison cannot tell "same 66581 chunks" from "66580 chunks
    plus one replacement", so verification done on a modified KB would be
    silently reused. We hash every chunk id AND its text: ids are assigned per
    write, so any re-write changes the digest even when the text is identical.
    """
    ids: list[str] = []
    texts: list[str] = []
    try:
        data = store._coll.get(include=["documents"])
        texts = list(data.get("documents") or [])
        ids = list(data.get("ids") or [])
    except Exception:
        pass
    h = hashlib.sha256()
    for i, t in sorted(zip(ids, texts)):
        h.update(i.encode())
        h.update(b"\x00")
        h.update((t or "").encode())
        h.update(b"\x01")
    return {"collection": store._coll.name,
            "total": store.count(),
            "poison": store.count(poison_only=True),
            "content_sha256": h.hexdigest()}


_CKPT_COND = "condition"   # key holding the run conditions a checkpoint belongs to


def _ckpt_load(path: str, condition: dict) -> dict:
    """Load resumable answers, but only if they were measured under `condition`.

    Reusing by qid alone would mix answers produced by a different model, KB or
    retrieval window into the current pool while the output claims the current
    conditions (review, 2026-10-26).
    """
    if not os.path.exists(path):
        return {}
    try:
        blob = json.load(open(path))
    except Exception as e:
        raise SystemExit(
            f"[dc] checkpoint {os.path.basename(path)} exists but cannot be read "
            f"({type(e).__name__}: {e}). Refusing to overwrite it: delete it "
            f"deliberately if the partial progress should be discarded.")
    stored = (blob.get("meta") or {}).get(_CKPT_COND)
    if stored != condition:
        raise SystemExit(
            f"[dc] checkpoint {os.path.basename(path)} was produced under different "
            f"conditions (stored={stored}, now={condition}). Delete it or pass a "
            f"different --out; mixing them would mislabel the pool.")
    return blob.get("answers") or {}


def _save_ckpt(path: str, answers: dict, condition: dict) -> None:
    """Write the FULL accumulated set (callers pass the merged dict)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"meta": {_CKPT_COND: condition,
                            "n_done": len(answers)}, "answers": answers}, fh)
    os.replace(tmp, path)


def load_questions(dataset: str, limit: int, seed: int):
    if dataset == "hotpot":
        from agentic_rag.data.hotpot import load_hotpot
        return load_hotpot(DATASETS[dataset]["corpus"], n_questions=limit, seed=seed)
    from agentic_rag.data.musique import load_musique
    return load_musique(DATASETS[dataset]["corpus"], n_questions=limit, seed=seed)


def ask_all(store, config, questions, model: str, label: str, workers: int,
             ckpt_path: str = "", ckpt_every: int = 200,
             ckpt_cond: dict | None = None, accumulated: dict | None = None):
    from concurrent.futures import ThreadPoolExecutor

    from agentic_rag.agents.base import SYSTEM_VICTIM, victim_system_prompt, ReActAgent, make_kb_search_tool
    from agentic_rag.llm import load_llm_config

    llm_cfg = load_llm_config(config, "agent")
    top_k = int(config["kb"].get("top_k", 8))
    n = max(1, min(workers, len(questions)))
    victims = [ReActAgent(name="victim", llm_cfg=llm_cfg, store=store,
                          system_prompt=victim_system_prompt(),
                          tools=[make_kb_search_tool(store, top_k)], top_k=top_k)
               for _ in range(n)]
    out: dict[str, dict] = dict(accumulated or {})
    n_err = 0

    # Instrument the RETRIEVER ITSELF while the victim runs. A post-hoc
    # `store.search` on the recorded query proves nothing about what the model
    # received: the query may have been malformed, the tool may have errored, or
    # the result set may have been empty (review round 4/5). Recording the real
    # calls is the only honest check.
    # ContextVar, not threading.local(): LangGraph may execute the tool in a
    # different thread, so thread-local storage loses the record even when the
    # retriever really returned passages (verified by review, round 6).
    from contextvars import ContextVar

    _calls: ContextVar[list | None] = ContextVar("dc_retrieval_calls", default=None)
    _orig_search = store.search

    def _instrumented_search(query, k=5, author_filter=None):
        try:
            out = _orig_search(query, k, author_filter)
        except Exception:
            raise
        calls = _calls.get()
        if calls is not None:
            try:
                n_chars = sum(len(getattr(c, "text", "") or "") for c in out)
            except Exception:
                n_chars = 0
            calls.append({"query": str(query)[:200], "k": k,
                          "n_results": len(out), "chars": n_chars})
        return out

    store.search = _instrumented_search

    def one(iq):
        i, q = iq
        rec: list = []
        _calls.set(rec)
        r = victims[i % n].ask(q.question)
        return q, r, rec

    with ThreadPoolExecutor(max_workers=n) as ex:
        for i, (q, r, calls) in enumerate(ex.map(one, enumerate(questions)), 1):
            pred = r.get("answer") or ""
            err = r.get("error")
            # "retrieved" = the retriever really returned passages to this model
            got_docs = any(c["n_results"] > 0 and c["chars"] > 0 for c in calls)
            if err:
                n_err += 1
                print(f"  [{label}] ERROR on {q.qid}: {err[:120]}")
            out[q.qid] = {
                "question": q.question, "gold": q.answer, "pred": pred,
                "correct": bool(unified.correct(pred, q.answer)),
                "empty": not pred.strip(),
                "n_tool_calls": r.get("n_tool_calls", 0),
                "retrieved": got_docs,
                "retrieval_calls": calls,
                "error": err,
            }
            if ckpt_path and (i % ckpt_every == 0):
                _save_ckpt(ckpt_path, out, ckpt_cond or {})
            if i % 100 == 0 or i == len(questions):
                print(f"  [{label}] {i}/{len(questions)} (errors so far: {n_err})")
    store.search = _orig_search
    if ckpt_path:
        _save_ckpt(ckpt_path, out, ckpt_cond or {})
    if n_err:
        # A failed call is NOT evidence about the question, so it must not enter
        # the pool. But aborting the whole pass over a handful of transport /
        # tool-format failures throws away thousands of good measurements (that
        # is exactly what happened on 2026-10-26: 4 harmony parse errors voided
        # 7401 results). Record them, exclude them downstream, keep going.
        print(f"  [{label}] {n_err}/{len(questions)} calls failed and are marked "
              f"unusable (excluded from the pool, run continues)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--mode", choices=["screen", "verify", "export-survivors"],
                    default="screen",
                    help="export-survivors = write the FULL all-models-correct "
                         "intersection of an already verified pool as its own "
                         "dataset (no model calls)")
    ap.add_argument("--screen-model", default="gpt-oss-20b")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--pool", type=int, default=500)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0,
                    help="screen only this many corpus questions (0 = all). The pool "
                         "is a random subset of clean-correct questions, so scanning "
                         "the whole corpus is wasteful: 1200 questions at ~50%% clean-"
                         "correct already yields ~600 candidates for --pool 500")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool-file", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=32,
                    help="concurrent victim asks; the vLLM server runs "
                         "--max-num-seqs 64, so 32 is safe and ~4x faster than 8")
    ap.add_argument("--one-model-per-run", type=int, default=1,
                    help="1 = this invocation verifies ONE already-served model and "
                         "merges into the pool file (driver loops over models). "
                         "0 = verify all --models in one go (requires a gateway that "
                         "really serves all of them; checked, else refused).")
    args = ap.parse_args()

    ds = DATASETS[args.dataset]
    config = load_config()
    config["kb"]["persist_dir"] = ds["kb_dir"]
    config["kb"]["collection"] = ds["kb_collection"]
    top_k = int(config["kb"].get("top_k", 8))
    base_url = config["llm"]["local"]["base_url"]
    store = get_store(config)

    fp0 = kb_fingerprint(store)
    if fp0["poison"]:
        raise SystemExit(
            f"[dc] {fp0['collection']} holds {fp0['poison']} poison chunks. D_c must "
            f"be measured on a clean KB. Clean it first.")
    if fp0["total"] < 1000:
        raise SystemExit(f"[dc] KB looks too small ({fp0['total']} chunks) — wrong "
                         f"persist_dir/collection?")
    left = store.delete_writer("doc-consolidator")
    if left:
        print(f"[dc] removed {left} leftover doc-consolidator chunks: D_c must be "
              f"measured on the SAME KB state the attack will start from "
              f"(longtail_attack also wipes them at start).")
    # fingerprint AFTER the cleanup: this is the state the attack really sees
    fp = kb_fingerprint(store)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    served = served_models(base_url)

    def served_name_for(requested: str, multi: bool) -> str:
        """Map a logical model name to the name the server actually hosts.

        A vLLM instance hosts ONE model; if several names are advertised, the
        request must still pick the one matching `requested`. Returning
        served[0] unconditionally would send all four names to the first model
        and record "one model answered four times" as "four models agree"
        (review round 2).
        """
        if requested in served:
            return requested
        if multi or len(served) > 1:
            raise SystemExit(
                f"[dc] '{requested}' is not hosted by {base_url} (it serves {served}). "
                f"Start the server for that model first.")
        # single-served alias: config name differs from --served-model-name
        return served[0]

    if args.mode == "screen":
        questions = load_questions(args.dataset, args.limit or 10**9, args.seed)
        want = args.screen_model
        if served and want not in served:
            raise SystemExit(
                f"[dc] server at {base_url} serves {served}, not '{want}'. Serve it "
                f"first (bash serve_model.sh {want}) — sending a name the server "
                f"does not host just fails every request.")
        config["llm"]["local"]["model"] = served_name_for(want, multi=False)
        ckpt = args.out.replace(".json", "_ckpt.json") if args.out else os.path.join(
            REPO, "data", "targets", f"dc_pool_{args.dataset}_ckpt.json")
        cond = {"model": config["llm"]["local"]["model"],
                "kb_sha": fp["content_sha256"], "window_k": top_k,
                "dataset": args.dataset,
                # every request-level setting that changes the victim's answer
                "temperature": config["llm"].get("temperature"),
                "max_tokens": config["llm"].get("max_tokens"),
                "payload_max_tokens": os.environ.get("AGENTIC_RAG_PAYLOAD_MAX_TOKENS"),
                "probe_max_tokens": os.environ.get("AGENTIC_RAG_PROBE_MAX_TOKENS"),
                "top_level": os.environ.get("AGENTIC_RAG_TOP_LEVEL_KWARGS"),
                "chat_kwargs": os.environ.get("AGENTIC_RAG_CHAT_KWARGS")}
        done = _ckpt_load(ckpt, cond)
        todo = [q for q in questions if q.qid not in done]
        print(f"[dc] screen: {len(questions)} {args.dataset} questions with "
              f"{served or want}, k={top_k}, KB={fp}")
        if done:
            print(f"[dc] resuming: {len(done)} already answered, {len(todo)} to go")
        recs = dict(done)
        if todo:
            recs = ask_all(store, config, todo, want, "screen", args.workers,
                           ckpt_path=ckpt, ckpt_cond=cond, accumulated=recs)

        # yes/no questions are unusable under substring scoring
        good = [q for q in questions
                if not recs[q.qid].get("error")          # measurement completed
                and recs[q.qid]["correct"]
                and unified.normalize(q.answer) not in YESNO
                and recs[q.qid]["retrieved"]]   # must actually retrieve passages
        print(f"[dc] clean-correct & usable: {len(good)}/{len(questions)}")
        order = list(range(len(good)))
        import random
        random.Random(args.seed).shuffle(order)
        chosen = [good[i] for i in order[:args.pool]]
        out = {
            "meta": {"dataset": args.dataset, "stage": "screen",
                     "served_models": served, "screen_model": want,
                     "n_screened": len(questions), "n_usable": len(good),
                     "pool_size": len(chosen), "seed": args.seed,
                     "window_k": top_k, "kb": fp,
                     "excluded_yesno": True,
                     "requirement": "n_tool_calls>0 (answer must come from retrieval)",
                     "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
            "candidates": [{"qid": q.qid, "question": q.question,
                            "gold": q.answer,
                            "supporting_titles": q.supporting_titles(),
                            "true_paragraph": q.true_paragraph_text() or "",
                            "topic": (q.supporting_titles() or ["unknown"])[0]}
                           for q in chosen],
            "screening": {q.qid: recs[q.qid] for q in chosen},
            "verified": {},
        }
        path = args.out or os.path.join(
            REPO, "data", "targets", f"dc_pool_{args.dataset}.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)
        print(f"[dc] pool of {len(chosen)} -> {path}")
        return

    if args.mode == "export-survivors":
        if not args.pool_file:
            raise SystemExit("[dc] export-survivors needs --pool-file")
        pool = json.load(open(args.pool_file))
        cands = pool["candidates"]
        models = pool["meta"].get("verified_models") or []
        verified = pool.get("verified") or {}
        missing = [m for m in DEFAULT_MODELS if m not in verified]
        if missing:
            raise SystemExit(f"[dc] pool is missing verification for {missing}")
        models = [m for m in DEFAULT_MODELS if m in verified]

        def q_sha(c):
            return hashlib.sha256((c["question"] + "\x00" + c["gold"]).encode()).hexdigest()

        keep = []
        for c in cands:
            sha = q_sha(c)
            if all(verified[m][c["qid"]]["correct"]
                   and verified[m][c["qid"]].get("retrieved")
                   and verified[m][c["qid"]].get("q_sha") == sha for m in models):
                keep.append(c)
        out = {c["qid"]: {"question": c["question"], "gold": c["gold"], "wrong": "",
                          "supporting_titles": c["supporting_titles"],
                          "true_paragraph": c["true_paragraph"],
                          "topic": c["topic"]} for c in keep}
        path = args.out or os.path.join(REPO, "data", "targets",
                                        f"dc_{args.dataset}_{len(out)}_all.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)
        body = json.dumps(out, sort_keys=True).encode()
        with open(path.replace(".json", ".meta.json"), "w") as fh:
            json.dump({
                "dataset": args.dataset, "n": len(out), "models": models,
                "window_k": pool["meta"]["window_k"], "kb": pool["meta"]["kb"],
                "pool_file": args.pool_file, "seed": pool["meta"].get("seed"),
                "criterion": ("every model answered correctly WITH retrieval on the "
                              "clean KB; superset of the frozen D_c"),
                "sha256_questions": hashlib.sha256(body).hexdigest(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, fh, indent=1)
        print(f"[dc] ALL-model-correct intersection: {len(out)} -> {path}")
        return

    # ---------------- verify ----------------
    if not args.pool_file:
        raise SystemExit("[dc] --mode verify needs --pool-file")
    pool = json.load(open(args.pool_file))
    if pool["meta"]["dataset"] != args.dataset:
        raise SystemExit(f"[dc] pool file is for '{pool['meta']['dataset']}', "
                         f"not '{args.dataset}'")
    if pool["meta"]["window_k"] != top_k:
        raise SystemExit(
            f"[dc] pool was screened with k={pool['meta']['window_k']}, now k={top_k}. "
            f"A set that is fully correct at one window size is not necessarily "
            f"correct at another; re-screen instead of mixing.")
    cands = pool["candidates"]
    if len({c["qid"] for c in cands}) != len(cands):
        raise SystemExit("[dc] pool file contains duplicate qids")

    if args.one_model_per_run:
        if not served:
            raise SystemExit(f"[dc] no model answering at {base_url}")
        if len(served) != 1:
            raise SystemExit(f"[dc] expected exactly one served model, got {served}")
        this_model = served[0]
        if this_model not in MODELS_EXPECTED_ALL(models):
            print(f"[dc] note: '{this_model}' is not one of the main-table victims")
        target_models = [this_model]
    else:
        missing = [m for m in models if m not in served]
        if missing:
            raise SystemExit(
                f"[dc] --one-model-per-run 0 needs a server hosting ALL of {models}; "
                f"{base_url} serves {served} (missing {missing}). Refusing: sending "
                f"four names to one server would record 'one model answered four "
                f"times' as 'four models agree'.")

        target_models = models

    class _Q:
        def __init__(self, rec):
            self.qid = rec["qid"]
            self.question = rec["question"]
            self.answer = rec["gold"]

        def supporting_titles(self):
            return []

    qs = [_Q(c) for c in cands]
    verified = pool.get("verified") or {}
    if verified:
        # a model verified against a different KB state is not comparable: the
        # same chunk count does not mean the same content (review round 2)
        stale = {m for m, rows in verified.items()
                 if {r.get("kb_sha") for r in rows.values()} - {fp["content_sha256"]}}
        if stale:
            raise SystemExit(
                f"[dc] verification for {sorted(stale)} was done on a different KB "
                f"({fp['total']} chunks now). Delete pool 'verified' or re-screen.")
        if pool["meta"].get("window_k") != top_k:
            raise SystemExit("[dc] window_k changed since verification; re-screen.")
    for m in target_models:
        config["llm"]["local"]["model"] = served_name_for(m, multi=not args.one_model_per_run)
        print(f"[dc] --- verifying with {m} (served as "
              f"{config['llm']['local']['model']}) ---")
        res = ask_all(store, config, qs, m, m, args.workers)
        run_cond = {"model": served_name_for(m, multi=not args.one_model_per_run),
                    "model_path": MODEL_PATHS.get(m, ""),
                    "served_models": served,
                    "window_k": top_k,
                    "payload_max_tokens": os.environ.get("AGENTIC_RAG_PAYLOAD_MAX_TOKENS"),
                    "probe_max_tokens": os.environ.get("AGENTIC_RAG_PROBE_MAX_TOKENS"),
                    "top_level": os.environ.get("AGENTIC_RAG_TOP_LEVEL_KWARGS"),
                    "chat_kwargs": os.environ.get("AGENTIC_RAG_CHAT_KWARGS"),
                    "kb_sha": fp["content_sha256"]}
        pool.setdefault("run_conditions", {})[m] = run_cond
        verified[m] = {q.qid: {"correct": res[q.qid]["correct"],
                               "pred": res[q.qid]["pred"],
                               "n_tool_calls": res[q.qid]["n_tool_calls"],
                               "retrieved": res[q.qid]["retrieved"],
                               "q_sha": hashlib.sha256(
                                   (q.question + "\x00" + q.answer).encode()
                               ).hexdigest(),
                               "kb_total": fp["total"],
                               "kb_sha": fp["content_sha256"]}
                       for q in qs}
        k = sum(1 for q in qs if res[q.qid]["correct"])
        k_ret = sum(1 for q in qs if res[q.qid]["correct"]
                    and res[q.qid]["retrieved"])
        print(f"[dc] {m}: clean-correct {k}/{len(qs)} "
              f"(with retrieval: {k_ret})")

    pool["verified"] = verified
    pool["meta"]["verified_models"] = sorted(verified)
    with open(args.pool_file, "w") as fh:
        json.dump(pool, fh, indent=1, ensure_ascii=False)

    still_missing = [m for m in models if m not in verified]
    if still_missing:
        print(f"[dc] still to verify: {still_missing}. Re-run with the server "
              f"serving each one. Nothing frozen yet.")
        return

    def _q_sha(c):
        return hashlib.sha256((c["question"] + "\x00" + c["gold"]).encode()).hexdigest()

    survivors = []
    for q in qs:
        c = next(c for c in cands if c["qid"] == q.qid)
        sha = _q_sha(c)
        ok = True
        for m in models:
            v = verified[m][q.qid]
            if not (v["correct"] and v.get("retrieved")):
                ok = False
                break
            # the stored verdict must belong to THIS question text and gold;
            # a changed gold would otherwise reuse a stale correct=True
            if v.get("q_sha") != sha:
                print(f"[dc] WARNING {q.qid}: verified text/gold changed for {m}; "
                      f"discarding that verdict")
                ok = False
                break
        if ok:
            survivors.append(q)
    print(f"[dc] survivors (ALL {len(models)} correct): {len(survivors)}/{len(qs)}")
    if len(survivors) < args.n:
        raise SystemExit(
            f"[dc] only {len(survivors)} survivors, need {args.n}. Re-run the screen "
            f"with a larger --pool (currently {len(cands)}). Nothing was frozen.")

    import random
    order = list(range(len(survivors)))
    random.Random(args.seed).shuffle(order)
    frozen = [survivors[i] for i in order[:args.n]]

    # final file: top level IS the qid map, which is what --target-records reads
    targets = {}
    for q in frozen:
        c = next(c for c in cands if c["qid"] == q.qid)
        targets[q.qid] = {
            "question": q.question,
            "gold": q.answer,
            # the wrong answer is filled by prepare_dc_targets.py (it needs an
            # LLM and a per-target wrong that is NOT a substring of gold)
            "wrong": "",
            "supporting_titles": c["supporting_titles"],
            "true_paragraph": c["true_paragraph"],
            "topic": c["topic"],
        }
    final = {qid: rec for qid, rec in targets.items()}
    path = args.out or os.path.join(
        REPO, "data", "targets", f"dc_{args.dataset}_{args.n}.json")
    with open(path, "w") as fh:
        json.dump(final, fh, indent=1, ensure_ascii=False)
    meta_path = path.replace(".json", ".meta.json")
    # Hash ONLY the immutable part (question text + gold). The `wrong` field is
    # filled in later by prepare_dc_targets.py, so including it would make the
    # freeze check fire on every legitimate re-run.
    body = json.dumps({q: {"question": r["question"], "gold": r["gold"]}
                       for q, r in final.items()}, sort_keys=True).encode()
    with open(meta_path, "w") as fh:
        json.dump({
            "dataset": args.dataset, "n": len(final), "models": models,
            "window_k": top_k, "kb": fp, "pool_file": args.pool_file,
            "seed": args.seed, "survivors": len(survivors),
            "sha256_questions": hashlib.sha256(body).hexdigest(),
            "served_models_seen": served,
            "criterion": ("every listed model answers correctly on the clean KB, "
                          "retrieval was used (n_tool_calls>0), and the gold answer "
                          "is not bare yes/no; flip denominator therefore equals "
                          f"|D_c|={len(final)} for all of them"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, fh, indent=1)
    print(f"[dc] FROZEN D_c {len(final)} -> {path}")
    print(f"[dc] meta -> {meta_path}")
    print(f"[dc] next: python prepare_dc_targets.py --dc {path}  (fills wrong answers)")


def MODELS_EXPECTED_ALL(models):
    return set(DEFAULT_MODELS)


if __name__ == "__main__":
    main()
