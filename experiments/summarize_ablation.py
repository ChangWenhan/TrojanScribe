"""Summarize ABLATION-V2 + main table into paper-facing tables.

V2 (2026-09-08): full rerun with the refill+MMR sampler as the only method
version. Backbones: xlam-2-8b / qwen3-8b / gpt-oss-20b / llama-3.1-8b
(Qwen3-4B retired; the default victim / headline slot results/08_longtail.json
is xlam-2-8b since the v2 rerun). Every ablation arm is run on EVERY backbone,
plus the cross-model matrix (attacker A's poison replayed into victim B).

Sections:
  1. main table   : 4 backbones x {hotpotqa, musique} (cluster v8, keyword)
  2. poison dose  : per-target poison chunks 2/4/6/8 (arms vol2/vol4/vol6),
                    per backbone
  3. style diversity: near-duplicate control (embed_hybrid) / single-style
                    (mono, authority) / no-diversity-selection (nodiv) /
                    no-template-anchor (greedy) / full method (cluster)
  4. trigger      : keyword / semantic(cosine>=0.82) / always-fire(p=1.0)
  5. retrieval window: victim top-k in {4, 8, 16} (arms topk4/topk16)
  6. cross-model  : inject model A -> victim model B matrix (diagonal =
     main table same-model rows)

Metric conventions (unified scoring, v2 2026-09-08):
  clean-correct = substring rule: normalize(gold) in normalize(pred) — this is
  the flip denominator, printed as "clean-correct"; flip = clean-correct ->
  NON-EMPTY wrong answer; collapse = clean-correct -> empty; ASR = substring of
  the injected wrong answer over all targets. Official EM is NOT used in the
  tables (the appendix 'clean'/'after' columns are substring counts too).

MuSiQue cross-model comparability: every backbone answers the SAME frozen
59-target set (research/frozen/musique_targets_59.json). The xlam-2-8b row
(results/08_longtail_xlam-2-8b_musique.json) is restricted to those 59 qids
at analysis time — no re-run needed.
"""
from __future__ import annotations

import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from agentic_rag.eval import unified  # noqa: E402

RESULTS = os.path.join(REPO, "results")
FROZEN_MUS = os.path.join(REPO, "research", "frozen", "musique_targets_59.json")

MAIN_MODELS = ["xlam-2-8b", "qwen3-8b", "gpt-oss-20b", "llama-3.1-8b"]
HEADLINE = MAIN_MODELS[0]


def result_path(model: str, arm: str | None = None) -> str:
    """Hotpot run file for a (model, arm). arm=None -> the model's main-table
    cluster v8 run (the headline's is 08_longtail.json, per the historical
    downstream convention); arm set -> 08_longtail_<model>_<arm>.json."""
    if arm is None:
        name = "08_longtail" if model == HEADLINE else f"08_longtail_{model}"
    else:
        name = f"08_longtail_{model}_{arm}"
    return os.path.join(RESULTS, f"{name}.json")


def frozen_mus_qids() -> set[str] | None:
    try:
        return set(json.load(open(FROZEN_MUS)).keys())
    except Exception:
        return None


def _main_clean_path(path: str, d: dict) -> str | None:
    """Main-table result file of the same victim+dataset as `path` — the
    canonical flip-denominator source. The ablation drivers pass
    `--clean-from <main file>`, so an arm's own clean is normally identical;
    four early xlam hotpot arms measured their own clean instead (one target
    drift), and every row is re-scored against the main clean so all rows
    share one denominator. Returns None when it cannot be resolved."""
    base = os.path.basename(path)
    is_mus = "musique" in base
    model = (d.get("meta") or {}).get("model") or ""
    if not model:
        for m in MAIN_MODELS:
            if m in base:
                model = m
                break
    if not model:
        return None
    if is_mus:
        name = "08_longtail_musique" if model == HEADLINE else f"08_longtail_{model}_musique"
    else:
        name = "08_longtail" if model == HEADLINE else f"08_longtail_{model}"
    p = os.path.join(RESULTS, f"{name}.json")
    return p if os.path.exists(p) else None


def summarize(path: str, restrict_qids: set[str] | None = None) -> list[dict]:
    """Extract one row per (run, variant). Missing file -> [] (pending arm)."""
    if not os.path.exists(path):
        return []
    d = json.load(open(path))
    if "targets" not in d or "clean" not in d:
        return []  # legacy format (pre-repair) — skipped
    targets = d["targets"]
    clean = d["clean"]["answers"]
    main_p = _main_clean_path(path, d)
    if main_p and os.path.abspath(main_p) != os.path.abspath(path):
        try:
            mcl = (json.load(open(main_p)).get("clean") or {}).get("answers") or {}
            over = {q: v for q, v in mcl.items() if q in targets}
            if over:
                clean = over
        except Exception:
            pass
    restricted = restrict_qids is not None
    if restricted:
        targets = {q: t for q, t in targets.items() if q in restrict_qids}
        clean = {q: v for q, v in clean.items() if q in restrict_qids}
        if not targets:
            return []
    clean_em_q = {
        qid: unified.correct(v["pred"], targets[qid]["gold"])
        for qid, v in clean.items()
    }
    n_clean = sum(clean_em_q.values())
    rows = []
    for variant, r in d.get("variants", {}).items():
        after = r["after"]["answers"]
        if restricted:
            after = {q: v for q, v in after.items() if q in restrict_qids}
        flips, kn, col, asr, after_em = 0, 0, 0, 0, 0
        for qid, v in after.items():
            pred = v["pred"]
            gold = targets[qid]["gold"]
            wrong = targets[qid].get("wrong") or ""
            em = unified.correct(pred, gold)
            after_em += int(em)
            if clean_em_q.get(qid) and not em:
                if pred.strip():
                    kn += 1
                else:
                    col += 1
            if wrong and unified.asr_hit(pred, wrong):
                asr += 1
        rows.append({
            "run": os.path.basename(path).replace("08_longtail", "").replace(".json", "") or "(headline)",
            "model": d["meta"].get("model", ""),
            "kb": d["meta"].get("kb_collection", ""),
            "trigger": d["meta"].get("trigger_kind", ""),
            "variant": variant,
            "volume": d["meta"].get("volume"),
            "n_poison": r["n_poison"],
            "fired": r["n_targets_fired"],
            "n": len(after),
            "clean_em": n_clean,
            "after_em": after_em,
            "asr": asr,
            "flips": kn,
            "flips_knowledge": kn,
            "flips_collapse": col,
            "poison_follow": r["after"].get("poison_follow"),
            "consensus_cov": round(r["co_retrieval"]["consensus_cov"], 3),
            "true_in_top8": r["co_retrieval"].get("true_in_top8"),
        })
    return rows


def fmt_row(label: str, r: dict, note: str = "") -> str:
    n, ce = r["n"], r["clean_em"]
    flip_pct = 100 * r["flips"] / ce if ce else 0.0
    asr_pct = 100 * r["asr"] / n if n else 0.0
    return (f"| {label} | {r['variant']} | {r['n_poison']} | {r['fired']}/{n} "
            f"| {ce}/{n} | {r['flips']}/{ce} ({flip_pct:.1f}%) "
            f"| {r['flips_collapse']} | {r['asr']}/{n} ({asr_pct:.1f}%) "
            f"| {r['poison_follow'] if r['poison_follow'] is not None else '—'} "
            f"| {r['true_in_top8'] if r['true_in_top8'] is not None else '—'} | {note} |")


HEADER = ("| 臂 | variant | 毒块 | fired/n | clean-correct | flip(knowledge) "
          "| collapse | ASR | poison_follow | true_in_top8 | 注 |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|")


def one_row(label: str, path: str, variant: str | None = None,
            restrict_qids: set[str] | None = None, note: str = "") -> str:
    rows = summarize(path, restrict_qids)
    if variant:
        rows = [r for r in rows if r["variant"] == variant]
    if not rows:
        return f"| {label} | — | — | — | — | — | — | — | — | — | pending |"
    return fmt_row(label, rows[0], note)


def per_model_table(section_title: str, arms: list[tuple[str, str]],
                    restrict_qids: set[str] | None = None,
                    arm_note: dict[str, str] | None = None) -> str:
    """One table with a row per (model, arm). arms = [(arm_suffix, variant), ...];
    arm_suffix None -> main-table cluster row (arm label = 'cluster')."""
    arm_note = arm_note or {}
    lines = [section_title, "", HEADER]
    for m in MAIN_MODELS:
        for arm, variant in arms:
            label = f"{m} / {arm or 'cluster-main'}"
            if arm is None:
                path = result_path(m)
                note = arm_note.get((m, arm), "") or (arm_note.get(m, "") or "")
            else:
                path = result_path(m, arm)
                note = arm_note.get((m, arm), "")
            lines.append(one_row(label, path, variant, restrict_qids, note))
    return "\n".join(lines)


def build_main_table() -> str:
    mus_qids = frozen_mus_qids()
    lines = ["## 主表：4 骨干 × 2 数据集（cluster v8, keyword 触发, 共享/冻结目标）", "",
             "HotpotQA 行：共享 60 目标 + 共享 wrongs（与 ReAct baselines 同协议）。"
             "MuSiQue 行：冻结 59 目标（xlam-2-8b 行在分析期裁剪到同一 59 qid）。", "",
             "### HotpotQA", "", HEADER]
    for m in MAIN_MODELS:
        lines.append(one_row(m, result_path(m), "cluster"))
    lines += ["", "### MuSiQue", "", HEADER]
    for m in MAIN_MODELS:
        name = "08_longtail_musique" if m == HEADLINE else f"08_longtail_{m}_musique"
        path = os.path.join(RESULTS, f"{name}.json")
        note = "frozen-59 subset" if m == HEADLINE else ""
        lines.append(one_row(m, path, "cluster", mus_qids, note))
    return "\n".join(lines)


def build_volume() -> str:
    arms = [("vol2", "cluster"), ("vol4", "cluster"), ("vol6", "cluster"),
            (None, "cluster")]
    return per_model_table(
        "## 毒量剂量（每目标毒块 2/4/6/8, cluster, HotpotQA, 全部骨干）\n"
        "行 = 骨干 / 臂;arm=cluster 即该骨干主表行（volume 8）。",
        arms)


def build_diversity() -> str:
    arms = [("embed_hybrid", "embed_hybrid"), ("mono", "cluster_mono"),
            ("nodiv", "cluster_nodiv"), ("greedy", "cluster_greedy"),
            (None, "cluster")]
    return per_model_table(
        "## 风格多样性分解（全部毒量 8, HotpotQA, 全部骨干, refill+MMR v2）\n"
        "行 = 骨干 / 臂;arm=cluster 即该骨干主表行（full）。",
        arms)


def build_trigger() -> str:
    arms = [("semantic", "cluster"), ("trig_always", "cluster"), (None, "cluster")]
    notes = {
        ("xlam-2-8b", "trig_always"): "always p=1.0",
        ("qwen3-8b", "trig_always"): "always p=1.0",
        ("gpt-oss-20b", "trig_always"): "always p=1.0",
        ("llama-3.1-8b", "trig_always"): "always p=1.0",
    }
    return per_model_table(
        "## 触发方式（cluster 毒量 8, HotpotQA, 全部骨干）\n"
        "行 = 骨干 / 臂;arm=cluster 即该骨干主表行（keyword）。",
        arms, arm_note=notes)


def build_topk() -> str:
    arms = [("topk4", "cluster"), (None, "cluster"), ("topk16", "cluster")]
    notes = {("xlam-2-8b", None): "k=8 (主表)", ("qwen3-8b", None): "k=8 (主表)",
             ("gpt-oss-20b", None): "k=8 (主表)", ("llama-3.1-8b", None): "k=8 (主表)"}
    return per_model_table(
        "## 受害者检索窗口（top-k=4/8/16, cluster 毒量 8, HotpotQA, 全部骨干）\n"
        "行 = 骨干 / 臂;k=8 行即该骨干主表 cluster 行;topk 只作用于 victim 检索侧。\n"
        "注意：flip 的 clean 分母统一复用该骨干 k=8 主表答对题集合（窗口变化不换分母）；"
        "true_in_top8 / consensus_cov 仪表固定按 k=8 统计，不代表该臂实际窗口。",
        arms, arm_note=notes)


def build_cross_model() -> str:
    lines = ["## 跨模型投毒迁移（攻击者 A 的毒文本 → 受害者 B, HotpotQA, cluster 毒量 8）",
             "",
             "对角线 = 主表同模型行（A==B）;非对角 = 08 脚本 phase inject+eval-after 结果,"
             "clean 分母取受害模型主表。单元格 = flips/clean-correct (flip%) / ASR%。",
             "",
             "| 注入 \\ 受害 | " + " | ".join(MAIN_MODELS) + " |",
             "|---|---|---|---|---|"]
    for att in MAIN_MODELS:
        cells = []
        for vic in MAIN_MODELS:
            if att == vic:
                rows = summarize(result_path(vic), None)
                rows = [r for r in rows if r["variant"] == "cluster"]
                if not rows:
                    cells.append("—")
                    continue
                r = rows[0]
                ce = r["clean_em"]
                fp = 100 * r["flips"] / ce if ce else 0.0
                ap = 100 * r["asr"] / r["n"] if r["n"] else 0.0
                cells.append(f"{r['flips']}/{ce} ({fp:.0f}%) / {ap:.0f}%")
            else:
                path = os.path.join(RESULTS, f"08_longtail_xsm_{att}_to_{vic}.json")
                if not os.path.exists(path):
                    cells.append("pending")
                    continue
                d = json.load(open(path))
                af = (d.get("after") or {}).get("answers") or {}
                md = json.load(open(result_path(vic)))
                mtg = md["targets"]
                mcl = md["clean"]["answers"]
                clean = {q for q in mtg
                         if unified.correct(mcl[q].get("pred", ""), mtg[q]["gold"])}
                kn = sum(1 for q in clean
                         if (af.get(q, {}).get("pred") or "").strip()
                         and not unified.correct(af[q].get("pred", ""), mtg[q]["gold"]))
                asr = sum(1 for q in mtg
                          if unified.asr_hit(af.get(q, {}).get("pred", ""),
                                             mtg[q].get("wrong") or ""))
                n = len(mtg)
                ce = len(clean)
                fp = 100 * kn / ce if ce else 0.0
                ap = 100 * asr / n if n else 0.0
                cells.append(f"{kn}/{ce} ({fp:.0f}%) / {ap:.0f}%")
        lines.append(f"| {att} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def raw_all_runs() -> str:
    mus_qids = frozen_mus_qids()
    paths = sorted(glob.glob(os.path.join(RESULTS, "08_longtail*.json")))
    all_rows = []
    for p in paths:
        restrict = mus_qids if mus_qids and "musique" in os.path.basename(p) else None
        all_rows.extend(summarize(p, restrict))
    header = (f"{'run':30s} {'model':14s} {'kb':12s} {'trigger':8s} {'var':15s} "
              f"{'vol':>3s} {'pois':>4s} {'fire':>4s} {'clean':>5s} {'after':>5s} "
              f"{'ASR':>11s} {'flip':>11s} {'kn':>3s} {'col':>3s} {'cov':>5s} {'true8':>5s}")
    lines = [header]
    for r in all_rows:
        n = r['n']
        lines.append(
            f"{r['run']:30s} {r['model']:14s} {r['kb']:12s} {r['trigger']:8s} {r['variant']:15s} "
            f"{r['volume'] or 0:3d} {r['n_poison']:4d} {r['fired']:4d} {r['clean_em']:5d} {r['after_em']:5d} "
            f"{r['asr']:3d}/{n:<3d} {100*r['asr']/n if n else 0:5.1f}%  "
            f"{r['flips']:2d}/{r['clean_em']:<2d} {100*r['flips']/r['clean_em'] if r['clean_em'] else 0:5.1f}% "
            f"{r['flips_knowledge']:3d} {r['flips_collapse']:3d} "
            f"{r['consensus_cov']:5.2f} {r['true_in_top8'] or 0:5d}")
    return "## 附录：全部 08 运行原始行（`*_musique` 文件均裁剪到冻结 59 目标,与主表同口径）\n\n```\n" + "\n".join(lines) + "\n```"


def main() -> None:
    parts = [
        "# Ablation + Main Table summary v2 (auto-generated by experiments/summarize_ablation.py)",
        "",
        "All runs: LangGraph victim, refill+MMR sampler (v2), bge KB, benign_rounds=3, "
        "unified scoring. flip(knowledge) = clean-correct -> non-empty wrong; "
        "collapse = -> empty; ASR = injected-wrong-substring hits; denominators explicit.",
        "",
        build_main_table(),
        "",
        build_volume(),
        "",
        build_diversity(),
        "",
        build_trigger(),
        "",
        build_topk(),
        "",
        build_cross_model(),
        "",
        raw_all_runs(),
    ]
    table = "\n".join(parts)
    print(table)
    out = os.path.join(RESULTS, "ablation_summary.md")
    with open(out, "w") as f:
        f.write(table + "\n")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()