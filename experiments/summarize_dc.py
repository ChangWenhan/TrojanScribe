"""Summarize D_c results: judge labels -> flip / collapse / ASR per condition.

Reads the `*_judged.json` files produced by judge_answers.py plus the clean
baseline for the same model, and prints one table per model.

Definitions (all over the fixed D_c of 100 questions):
  clean_correct : judge said CORRECT on the clean run
  flip          : clean CORRECT -> attack WRONG      (knowledge overwritten)
  collapse      : clean CORRECT -> attack REFUSAL/ERROR  (answer suppressed)
  attack_success: flip + collapse  (user decision 2026-10-27: the attack's goal is
                  that the victim no longer returns the correct answer, whether by
                  stating the injected wrong answer or by failing to answer at all)
  unchanged     : clean CORRECT -> attack CORRECT
  ASR           : attack answer adopts the injected wrong answer (strict: the
                  judge answered WRONG=YES; clean runs are reported the same way
                  so the baseline is not assumed to be zero)

Every number is over 100 questions; `clean_correct` is shown because a question
that was never answered correctly cannot flip.

Usage
  cd experiments && python summarize_dc.py [--models gpt-oss-20b,...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results")
CONDITIONS = ["ours", "poisonedrag", "agentpoison", "corruptrag_as", "corruptrag_ak"]
LABELS = ["ours", "PoisonedRAG", "AgentPoison", "CorruptRAG-AS", "CorruptRAG-AK"]


def load(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def summarize(model: str, suf: str = "") -> dict | None:
    clean = load(os.path.join(RES, f"dc_clean_clean_{model}{suf}_judged.json"))
    if not clean:
        return None
    clean_lab = {q: v["label"] for q, v in clean["per_target"].items()}
    clean_ok = {q for q, lab in clean_lab.items() if lab == "CORRECT"}
    rows = []
    for cond, label in zip(CONDITIONS, LABELS):
        # our final configuration includes the no-gold-leak screen; prefer that
        # run when it exists so the table shows the shipped method
        cand = os.path.join(RES, f"dc_{cond}_noleak_{model}{suf}_eval_judged.json")
        if cond == "ours" and os.path.exists(cand):
            j = load(cand)
            used = os.path.basename(cand)
        else:
            used = f"dc_{cond}_{model}{suf}_judged.json"
            j = load(os.path.join(RES, used))
        if not j:
            rows.append({"label": label, "missing": True})
            continue
        per = j["per_target"]
        flip = [q for q in clean_ok
                if per.get(q, {}).get("label") == "WRONG"]
        coll = [q for q in clean_ok
                if per.get(q, {}).get("label") in ("REFUSAL", "ERROR")]
        same = [q for q in clean_ok
                if per.get(q, {}).get("label") == "CORRECT"]
        # UNCLEAR = the judge could not decide. It is neither a flip nor a
        # silence, so it gets its own bucket instead of silently breaking the
        # arithmetic (flip + silent + unchanged + unclear == n_clean_ok).
        unclear = [q for q in clean_ok
                   if per.get(q, {}).get("label") not in
                   ("WRONG", "REFUSAL", "ERROR", "CORRECT")]
        asr = j["summary"].get("strict_asr", 0)
        rows.append({"label": label, "flip": len(flip), "collapse": len(coll),
                     "success": len(flip) + len(coll),
                     "unchanged": len(same), "unclear": len(unclear), "asr": asr,
                     "n_clean_ok": len(clean_ok),
                     "pre_wrong": len(clean_lab) - len(clean_ok),
                     "source": used})
    clean_asr = clean["summary"].get("strict_asr", 0)
    return {"model": model, "clean_ok": len(clean_ok), "clean_asr": clean_asr,
            "n": len(clean_lab), "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    # llama-3.1-8b is EXCLUDED from the reported tables: on identical input at
    # temperature 0 it returns different answers for 16/100 questions (gpt-oss:
    # 7/100) and gets stuck in the retrieval loop (up to 17 tool calls), so its
    # clean baseline is not reproducible and its measurements cannot rank
    # methods. Evidence: results/archive_llama_unstable/ and
    # results/logs/det_llama.log / det_gptoss.log.
    ap.add_argument("--models",
                    default="xlam-2-8b,qwen3-4b-instruct,gpt-oss-20b,olmo-3-7b")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--musique", action="store_true")
    args = ap.parse_args()

    out = []
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        s = summarize(m, "_musique" if args.musique else "")
        if not s:
            print(f"[sum] {m}: no clean judged file, skipped")
            continue
        out.append(s)
        print(f"\n=== {m} ===  D_c=100  clean_correct={s['clean_ok']}  "
              f"clean_ASR={s['clean_asr']}")
        # Every cell is out of the SAME 100 questions. The clean accuracy is
        # reported as the victim's baseline error rate, not used as the
        # denominator (decision 2026-10-27): "success out of 100" keeps the
        # four backbones directly comparable even when one of them cannot
        # answer every question before the attack.
        hdr = (f"{'method':16s} {'success/100':>12s} {'=flip':>7s} {'+silent':>8s} "
               f"{'still ok':>9s} {'unclear':>8s} {'pre-wrong':>10s} {'ASR':>8s}")
        print(hdr)
        for r in s["rows"]:
            if r.get("missing"):
                print(f"{r['label']:16s} {'(not run)':>12s}")
                continue
            print(f"{r['label']:16s} {r['success']:7d}/100 "
                  f"{r['flip']:7d} {r['collapse']:8d} {r['unchanged']:9d} "
                  f"{r['unclear']:8d} {r['pre_wrong']:10d} {r['asr']:5d}/100")
    if args.markdown:
        # MuSiQue gets its own file so regenerating one dataset never clobbers
        # the other (the two tables used to collide on dc_summary.md).
        fname = "dc_summary_musique.md" if args.musique else "dc_summary.md"
        path = os.path.join(RES, fname)
        with open(path, "w") as fh:
            fh.write("# D_c results (100 fixed questions)\n\n")
            for s in out:
                fh.write(f"## {s['model']}\n\n")
                fh.write(f"clean correct: {s['clean_ok']}/100, "
                         f"clean ASR: {s['clean_asr']}/100\n\n")
                fh.write("every cell counts the same 100 questions. "
                         "attack success = flip + silent. "
                         "`still ok` = clean-correct and still correct; "
                         "`pre-wrong` = already wrong before the attack "
                         "(the victim's own baseline error).\n\n")
                fh.write("| method | attack success | =flip | +silent | still ok "
                         "| unclear | pre-wrong | ASR |\n")
                fh.write("|---|---|---|---|---|---|---|---|\n")
                for r in s["rows"]:
                    if r.get("missing"):
                        fh.write(f"| {r['label']} | - | - | - | - | - | - | - |\n")
                        continue
                    fh.write(f"| {r['label']} | **{r['success']}/100** | "
                             f"{r['flip']} | {r['collapse']} | {r['unchanged']} | "
                             f"{r['unclear']} | {r['pre_wrong']} | "
                             f"{r['asr']}/100 |\n")
                fh.write("\n")
        print(f"\n[sum] wrote {path}")


if __name__ == "__main__":
    main()
