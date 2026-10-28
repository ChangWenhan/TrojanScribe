"""Read-only audit of the D_c ablation artifacts (no Chroma, no GPU, no writes).

The ablation table is only as good as the files behind it, and every way those
files can go wrong is invisible once the numbers are in Markdown:

  * a dose arm whose poison file actually holds 1 text per question
  * an eval that replayed an OLD version of the poison file (regenerated later)
  * an eval run on the wrong dataset / window / snapshot
  * a flip reference (clean run) measured on a different snapshot
  * a judged file that covers only part of the 100 questions
  * poison texts that state the CORRECT answer as a live fact

The last one is not a bug by itself: the embed_hybrid arm is built without the
"do not leak the gold" filter that the main method uses, so the number has to be
MEASURED and reported instead of assumed to be zero.

Usage
  cd experiments && python audit_ablation.py            # full report
  cd experiments && python audit_ablation.py --quiet    # failures only

Exit code 1 if a hard check fails (the run would be reported wrongly), 0 if
everything that exists is consistent.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from agentic_rag.agents.poisoned.payload import PayloadGenerator  # noqa: E402

RES = os.path.join(REPO, "results")
MODELS = ["xlam-2-8b", "qwen3-4b-instruct", "gpt-oss-20b", "olmo-3-7b"]
DATASETS = [("hotpot", ""), ("musique", "_musique")]
SETTINGS = ["vol2", "vol4", "vol6", "embed_hybrid", "mono", "nodiv", "greedy",
            "trig_semantic", "topk4", "topk16"]
DOSE = {"vol2": 2, "vol4": 4, "vol6": 6}
WINDOW = {"topk4": 4, "topk16": 16}
N = 100


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def expected_dose(setting: str) -> int:
    return DOSE.get(setting, 8)


# corpus size implied by each eval run (kb_total - snapshot - poison); must be
# identical for every run of the same dataset
KNOWN_CORPUS: dict[str, set] = {"hotpot": set(), "musique": set()}


def drain(blob) -> collections.Counter:
    """(qid, text) multiset of the poison writes stored in a poison/eval file."""
    out = collections.Counter()
    for v in (blob.get("variants") or {}).values():
        for w in v.get("poison_writes") or []:
            out[(w.get("qid"), w.get("text"))] += 1
    if not out:
        for w in blob.get("poison_writes") or []:
            out[(w.get("qid"), w.get("text"))] += 1
    return out


def audit_poison(setting, model, suf, ds, fails):
    pf = os.path.join(RES, f"dc_ours_{setting}_{model}{suf}.json")
    blob = load(pf)
    if blob is None:
        return None
    meta = blob.get("meta") or {}
    tags = []
    if meta.get("dc_file") and meta["dc_file"] != f"dc_{ds}_100.json":
        fails.append(f"{ds}/{setting}/{model}: poison built for {meta['dc_file']}")
        tags.append("WRONG_DC")
    if meta.get("model") and meta["model"] != model:
        fails.append(f"{ds}/{setting}/{model}: poison meta.model={meta['model']}")
        tags.append("WRONG_MODEL")
    if setting != "trig_semantic" and not meta.get("no_gold_leak"):
        fails.append(f"{ds}/{setting}/{model}: poison generated WITHOUT no-gold-leak")
        tags.append("NO_LEAKGATE")
    per_q = collections.Counter(q for v in (blob.get("variants") or {}).values()
                                for q in (w.get("qid") for w in v.get("poison_writes") or []))
    want = expected_dose(setting)
    if setting == "trig_semantic":
        if not per_q:
            fails.append(f"{ds}/{setting}/{model}: filtered poison is empty")
            tags.append("EMPTY")
        bad = {q: c for q, c in per_q.items() if c != want}
        if bad:
            fails.append(f"{ds}/{setting}/{model}: partial arm with off-dose questions "
                         f"{list(bad.items())[:3]}")
            tags.append("DOSE")
        coverage = len(per_q)
    else:
        # A generator may record documented shortfalls (meta.short_qids): those
        # questions could not reach the full dose because the frozen D_c "wrong"
        # string is dirty. Everything else must still be exactly `want`.
        short = meta.get("short_qids") or {}
        bad = {q: c for q, c in per_q.items() if q not in short and c != want}
        unlisted = [q for q in per_q if q in short and per_q[q] != short[q]]
        zero = [q for q, c in short.items() if c == 0 and q in per_q]
        if len(per_q) + len([q for q, c in short.items() if c == 0]) != N or bad \
                or unlisted or zero:
            fails.append(f"{ds}/{setting}/{model}: dose {want}/question violated "
                         f"({len(per_q)} questions, e.g. {list(bad.items())[:3]})")
            tags.append("DOSE")
        elif short:
            tags.append(f"short={len(short)}@mean{meta.get('mean_dose')}")
        coverage = len(per_q)
    return blob, ("" if not tags else ",".join(tags)), coverage, drain(blob)


def audit_eval(setting, model, suf, ds, pf_drain, fails):
    ef = os.path.join(RES, f"dc_ours_{setting}_{model}{suf}_eval.json")
    blob = load(ef)
    if blob is None:
        return "no eval"
    meta = blob.get("meta") or {}
    tags = []
    if meta.get("dataset") != ds:
        fails.append(f"{ds}/{setting}/{model}: eval meta.dataset={meta.get('dataset')}")
        tags.append("WRONG_DS")
    n_ans = len(blob.get("answers") or {})
    if n_ans != N:
        fails.append(f"{ds}/{setting}/{model}: eval has {n_ans}/100 answers")
        tags.append(f"N={n_ans}")
    want_k = WINDOW.get(setting, 8)
    if meta.get("window_k") != want_k:
        fails.append(f"{ds}/{setting}/{model}: eval window_k={meta.get('window_k')} "
                     f"!= {want_k}")
        tags.append("WINDOW")
    snap = load(os.path.join(REPO, "data", "targets", f"snapshot_{ds}_{model}.json"))
    snap_sha = ((snap or {}).get("meta") or {}).get("sha256_snapshot")
    if snap_sha and meta.get("snapshot_sha256") != snap_sha:
        fails.append(f"{ds}/{setting}/{model}: eval used a different snapshot")
        tags.append("SNAPSHOT")
    if pf_drain is not None and drain(blob) != pf_drain:
        fails.append(f"{ds}/{setting}/{model}: eval replayed different texts than the "
                     f"poison file on disk")
        tags.append("STALE_POISON")
    summ = blob.get("summary") or {}
    cf = summ.get("clean_file")
    if cf:
        cb = load(cf)
        csha = ((cb or {}).get("meta") or {}).get("snapshot_sha256")
        if csha and snap_sha and csha != snap_sha:
            fails.append(f"{ds}/{setting}/{model}: flip reference on another snapshot")
            tags.append("CLEAN_REF")
        if setting in WINDOW and os.path.basename(cf) != \
                f"dc_clean_clean_{model}{suf}_topk{WINDOW[setting]}.json":
            fails.append(f"{ds}/{setting}/{model}: window arm used {os.path.basename(cf)} "
                         f"as its flip reference")
            tags.append("CLEAN_K")
    errs = sum(1 for a in (blob.get("answers") or {}).values() if a.get("error"))
    if errs:
        tags.append(f"errors={errs}")
    # INVARIANT: kb_total - snapshot rows - poison rows = the untouched corpus.
    # Every run of the same dataset must show the same number; a leftover row
    # from another writer (or a missing corpus chunk) shows up here.
    try:
        corpus_n = (meta.get("kb_total") or 0) - (meta.get("n_snapshot_rows") or 0) \
            - (meta.get("n_poison_rows") or 0)
    except TypeError:
        corpus_n = None
    if corpus_n is not None:
        KNOWN_CORPUS[ds].add(corpus_n)
    return ",".join(tags) or "ok"


def audit_judged(setting, model, suf, ds, fails):
    jf = os.path.join(RES, f"dc_ours_{setting}_{model}{suf}_eval_judged.json")
    blob = load(jf)
    if blob is None:
        return "no judge"
    pt = blob.get("per_target") or {}
    if len(pt) != N:
        fails.append(f"{ds}/{setting}/{model}: judged file covers {len(pt)}/100")
        return "COVERAGE"
    labs = collections.Counter(v.get("label") for v in pt.values())
    return " ".join(f"{k}={v}" for k, v in sorted(labs.items()))


def gold_leaks(blob, ds):
    """Texts that state the CORRECT answer as a live fact (measured, not assumed)."""
    dc = load(os.path.join(REPO, "data", "targets", f"dc_{ds}_100.json")) or {}
    n = 0
    for v in (blob.get("variants") or {}).values():
        for w in v.get("poison_writes") or []:
            rec = dc.get(w.get("qid")) or {}
            if rec.get("gold") and PayloadGenerator._leaks_gold(
                    w.get("text") or "", rec["gold"], question=rec.get("question", "")):
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    fails: list[str] = []
    for ds, suf in DATASETS:
        print(f"\n=== {ds} ===")
        for model in MODELS:
            for setting in SETTINGS:
                got = audit_poison(setting, model, suf, ds, fails)
                if got is None:
                    continue
                blob, ptags, coverage, pf_drain = got
                etags = audit_eval(setting, model, suf, ds, pf_drain, fails)
                jtag = audit_judged(setting, model, suf, ds, fails)
                if args.quiet and "ok" in etags and jtag != "no judge":
                    continue
                leak = gold_leaks(blob, ds)
                print(f"{model:18s} {setting:14s} n_q={coverage:3d} "
                      f"gold_leak={leak:3d} poison[{ptags or 'ok'}] "
                      f"eval[{etags}] judge[{jtag}]")
    for ds in KNOWN_CORPUS:
        seen = KNOWN_CORPUS[ds]
        if len(seen) > 1:
            fails.append(f"{ds}: the untouched corpus size differs between runs "
                         f"{sorted(seen)} - leftover rows or a missing chunk")
        elif seen:
            print(f"{ds}: corpus+basis rows per run = {seen.pop()} (constant)")
    print()
    if fails:
        print(f"HARD FAILURES ({len(fails)}):")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("audit: every artifact that exists is consistent "
          "(files not generated yet are simply absent).")


if __name__ == "__main__":
    main()
