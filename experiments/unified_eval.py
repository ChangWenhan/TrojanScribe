"""Experiment 13: unified evaluation across the two victim frameworks.

Design (post-refactor, per project decision):
  - KidnapRAG methods (naive / poisonedRAG / ours / topicattack) run inside the
    KidnapRAG repo on THEIR OWN ReAct agent. Results live in
    kidnaprag/ReAct/results/adv_targeted_results/hotpotqa_seed1_<method>_qwen34.json
  - OUR method (consensus-cluster) runs on OUR langgraph agent via
    experiments/longtail_attack.py. Results live in results/08_longtail.json
  - The two frameworks are deliberately NOT merged: each attack is evaluated
    on the victim it was designed for.

This module re-scores BOTH formats with ONE identical set of functions (now
imported from src/agentic_rag/eval/unified.py) so that every reported number
(EM / F1>0 / ASR / flip-rate / empty) is comparable across frameworks.

REPAIR 2026-09-06:
  - the langgraph rows now take gold AND wrong from the per-target records
    persisted by 08 (wrong == the shared hotpotqa.json "incorrect answer"
    actually injected on both sides). Previously the langgraph ASR was scored
    against the shared wrong answers while the run had injected its own
    LLM-generated ones — the ASR column compared different target strings.
  - flips are decomposed into flip_knowledge (non-empty wrong answer) and
    flip_collapse (empty answer / crashed row): DoS-style agent breakage is
    never counted as a knowledge flip.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from agentic_rag.eval import unified  # noqa: E402  (single scoring source)

REACT_RESULTS = os.path.join(
    REPO, "kidnaprag", "ReAct", "results", "adv_targeted_results"
)
# v2 (2026-09-08): baselines re-run on GLM-4-9B; suffix derived from the served
# model name (attack_react.py: qwen3-4b -> qwen34, xlam-2-8b -> xlam28b)
M_SUFFIX = "xlam28b"
normalize = unified.normalize
f1_tokens = unified.f1_tokens
asr_hit = unified.asr_hit
score = unified.score


def load_react(method: str) -> dict:
    """KidnapRAG ReAct result JSON -> per-target rows (raw texts)."""
    path = os.path.join(REACT_RESULTS, f"hotpotqa_seed1_{method}_{M_SUFFIX}.json")
    d = json.load(open(path))
    rows = []
    for it in d["details"]:
        if it.get("qid") is None:
            continue
        rows.append({
            "qid": it["qid"],
            "answer": it.get("llm_answer", "") or "",
            "gold": it.get("correct_answer", "") or "",
            "wrong": it.get("target_answer", "") or "",
            "error": it.get("error", ""),
        })
    return {"framework": "react", "method": method, "rows": rows}


def load_langgraph() -> dict:
    """Our langgraph experiment (08_longtail.json) -> per-target rows.

    New format (repair 2026-09-06): gold/wrong come from the per-target
    records persisted by 08; clean answers from `clean.answers`, attacked
    answers from `variants.cluster.after.answers`. The legacy flat format
    (base_answers/after_answers + shared-file join) is still supported for
    archived results.
    """
    path = os.path.join(REPO, "results", "08_longtail.json")
    d = json.load(open(path))

    if "targets" in d and "variants" in d:
        targets = d["targets"]
        clean = d.get("clean", {}).get("answers", {})
        attack = d["variants"]["cluster"]["after"]["answers"]
        clean_rows = [
            {
                "qid": qid,
                "answer": v.get("pred", "") or "",
                "gold": targets.get(qid, {}).get("gold", ""),
                "wrong": targets.get(qid, {}).get("wrong", ""),
                "error": "",
            }
            for qid, v in clean.items()
        ]
        attack_rows = [
            {
                "qid": qid,
                "answer": v.get("pred", "") or "",
                "gold": targets.get(qid, {}).get("gold", ""),
                "wrong": targets.get(qid, {}).get("wrong", ""),
                "error": "",
            }
            for qid, v in attack.items()
        ]
        return {
            "framework": "langgraph",
            "method": "cluster",
            "clean_rows": clean_rows,
            "rows": attack_rows,
        }

    # legacy format (pre-repair archived results)
    shared = json.load(open(os.path.join(REACT_RESULTS, "hotpotqa.json")))
    base = d.get("base_answers", {})
    after = d.get("after_answers", {})
    clean_rows, attack_rows = [], []
    for qid, v in base.items():
        ref = shared.get(qid, {})
        clean_rows.append({
            "qid": qid,
            "answer": v.get("pred", "") or "",
            "gold": ref.get("correct answer", ref.get("answer", "")) or "",
            "wrong": ref.get("incorrect answer", ref.get("wrong_answer", "")) or "",
            "error": "",
        })
    for qid, v in after.items():
        ref = shared.get(qid, {})
        attack_rows.append({
            "qid": qid,
            "answer": v.get("pred", "") or "",
            "gold": ref.get("correct answer", ref.get("answer", "")) or "",
            "wrong": ref.get("incorrect answer", ref.get("wrong_answer", "")) or "",
            "error": "",
        })
    return {
        "framework": "langgraph",
        "method": "cluster",
        "clean_rows": clean_rows,
        "rows": attack_rows,
    }


def summarize(framework: str, method: str, clean_rows, attack_rows) -> dict:
    clean_sc = [score(r["answer"], r["gold"], r["wrong"]) for r in clean_rows]
    att_sc = [score(r["answer"], r["gold"], r["wrong"]) for r in attack_rows]
    clean_em_q = {r["qid"]: unified.correct(r["answer"], r["gold"]) for r in clean_rows}
    out = {
        "framework": framework,
        "method": method,
        "n": len(attack_rows),
        "clean_em": sum(1 for s in clean_sc if s["em"]),
        "after_em": sum(1 for s in att_sc if s["em"]),
        "after_f1pos": sum(1 for s in att_sc if s["f1_pos"]),
        "asr": sum(1 for s in att_sc if s["asr"]),
        "empty": sum(1 for s in att_sc if s["empty"]),
        "errors": sum(1 for r in attack_rows if r["error"]),
    }
    # flip := clean-EM-correct -> after NON-EMPTY wrong answer (knowledge was
    # actually rewritten: 本来真 -> 变成假). Empty answers / crashed rows are
    # NOT a knowledge flip — they are reported separately as `collapse`
    # (agent disabled). outcome_changed = flip + collapse = "the system's
    # answer changed from correct to something else", kept for reference.
    flips, collapse, outcome_changed = 0, 0, 0
    for r, s in zip(attack_rows, att_sc):
        if clean_em_q.get(r["qid"]) and not unified.correct(r["answer"], r["gold"]):
            outcome_changed += 1
            if r["error"] or s["empty"]:
                collapse += 1
            else:
                flips += 1
    out["flips"] = flips
    out["flips_collapse"] = collapse
    out["outcome_changed"] = outcome_changed
    out["flip_rate"] = round(flips / out["clean_em"], 3) if out["clean_em"] else None
    out["asr_rate"] = round(out["asr"] / out["n"], 3) if out["n"] else None
    out["after_em_rate"] = round(out["after_em"] / out["n"], 3) if out["n"] else None
    return out


METHODS = ["naive", "poisonedRAG", "ours", "topicattack"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "13_unified_comparison.json"))
    args = ap.parse_args()

    clean_r = load_react("clean")
    rows_all = [summarize("react", "clean", clean_r["rows"], clean_r["rows"])]

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        r = load_react(m)
        rows_all.append(summarize("react", m, clean_r["rows"], r["rows"]))

    lg = load_langgraph()
    rows_all.append(summarize("langgraph", lg["method"],
                              lg["clean_rows"], lg["rows"]))

    json.dump({"per_framework_clean_note": (
        "flip denominators are framework-local: react methods flip against the "
        "ReAct clean baseline, the langgraph method against the langgraph clean "
        "baseline. All metrics use identical scoring functions "
        "(src/agentic_rag/eval/unified.py). DEFINITION (2026-09-06): flip counts "
        "ONLY clean-EM-correct targets answered with a NON-EMPTY wrong answer "
        "(knowledge actually rewritten); empty/crashed rows are `collapse`, and "
        "outcome_changed = flip + collapse (answer changed from correct to "
        "anything else, including nothing)."),
        "rows": rows_all},
        open(args.out, "w"), indent=2)

    print(f"{'method':12s} {'fw':10s} {'n':>3s} {'EM0':>4s} {'EM1':>4s} "
          f"{'F1>0':>4s} {'ASR':>4s} {'ASR%':>5s} {'flip':>7s} {'flip%':>6s} "
          f"{'collapse':>8s} {'changed':>7s} {'empty':>5s} {'err':>4s}")
    for row in rows_all:
        print(f"{row['method']:12s} {row['framework']:10s} {row['n']:3d} "
              f"{row['clean_em']:4d} {row['after_em']:4d} {row['after_f1pos']:4d} "
              f"{row['asr']:4d} {100 * (row['asr_rate'] or 0):5.1f} "
              f"{row['flips']:2d}/{row['clean_em']:<2d} {100 * (row['flip_rate'] or 0):6.1f} "
              f"{row['flips_collapse']:8d} {row['outcome_changed']:7d} "
              f"{row['empty']:5d} {row['errors']:4d}")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
