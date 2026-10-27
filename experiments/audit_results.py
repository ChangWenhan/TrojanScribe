"""Read-only audit of raw answers, paper tables, and files in the Git release.

Run from experiments/ with the agents Python. Optional --csv saves audit
evidence outside results/; experiment JSONs and paper summaries are never edited.
The scoring rule is unchanged: this checks arithmetic, not semantic correctness.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from agentic_rag.eval import unified  # noqa: E402
import summarize_ablation as summary  # noqa: E402

MODELS = summary.MAIN_MODELS


def read(path):
    return json.loads(Path(path).read_text())


def main_path(model, musique):
    stem = "08_longtail" if model == MODELS[0] else f"08_longtail_{model}"
    return REPO / "results" / (stem + ("_musique" if musique else "") + ".json")


def score(targets, clean, after):
    correct = {q for q, t in targets.items()
               if unified.correct(clean[q]["pred"], t["gold"])}
    return {
        "n": len(targets), "clean": len(correct),
        "flip": sum(bool(after[q]["pred"].strip())
                    and not unified.correct(after[q]["pred"], targets[q]["gold"])
                    for q in correct),
        "collapse": sum(not after[q]["pred"].strip() for q in correct),
        "asr": sum(unified.asr_hit(after[q]["pred"], t["wrong"])
                   for q, t in targets.items()),
        "correct_and_asr": sum(unified.correct(after[q]["pred"], t["gold"])
                               and unified.asr_hit(after[q]["pred"], t["wrong"])
                               for q, t in targets.items()),
    }


def method_name(filename):
    if filename.startswith("08_longtail_xsm_"):
        return "transfer"
    if filename.startswith("08_longtail"):
        return "TS"
    if "corruptrag_as" in filename:
        return "CR-AS"
    if "corruptrag_ak" in filename:
        return "CR-AK"
    if "agentpoison" in filename:
        return "AP"
    return "PR"


def check_tables(path, rows, transfers, errors):
    """Check numeric cells in main/baseline/cross-model README tables."""
    headers, musique, checked = [], False, 0
    methods = {"PR": "PR", "PoisonedRAG": "PR", "AP": "AP",
               "AgentPoison": "AP", "CR-AS": "CR-AS", "CR-AK": "CR-AK",
               "TrojanScribe": "TS", "TS": "TS"}
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if line.startswith("#"):
            if "MuSiQue" in line:
                musique = True
            elif "HotpotQA" in line or "Baseline comparison" in line or "Cross-model" in line:
                musique = False
        if not line.startswith("|"):
            continue
        cells = [s.strip().replace("**", "") for s in line.strip("|").split("|")]
        first = cells[0].lower()
        if first in {"victim", "backbone"} or first.startswith("attacker "):
            headers = cells
            continue
        model = next((m for m in MODELS if first.startswith(m)), None)
        if model is None or not headers:
            continue
        if headers[0].lower().startswith("attacker "):
            for victim, cell in zip(MODELS, cells[1:]):
                if model == victim:
                    continue
                row = transfers[model, victim]
                match = re.search(r"(\d+)/(\d+)\s*\((\d+)%\).*?ASR (\d+)%", cell)
                expected = (row["flip"], row["clean"],
                            int(f"{100 * row['flip'] / row['clean']:.0f}"),
                            int(f"{100 * row['asr'] / row['n']:.0f}"))
                if not match or tuple(map(int, match.groups())) != expected:
                    errors.append(f"{path.relative_to(REPO)}:{line_no}: {cell} != {expected}")
                collapse = re.search(r"coll (\d+)", cell)
                if collapse and int(collapse[1]) != row["collapse"]:
                    errors.append(f"{path}:{line_no}: collapse differs")
                checked += 1
            continue
        for header, cell in zip(headers[1:], cells[1:]):
            method = methods.get(header.split()[0], "TS")
            row = rows[musique, model, method]
            if header.startswith("clean"):
                expected = row["clean"], row["n"]
            elif header == "collapse":
                if int(cell) != row["collapse"]:
                    errors.append(f"{path}:{line_no}: collapse differs")
                checked += 1
                continue
            elif "true paragraph" in header:
                expected = row["true_in_top8"], row["n"]
            elif "ASR" in header:
                expected = row["asr"], row["n"]
            elif "flip" in header:
                expected = row["flip"], row["clean"]
            else:
                continue
            match = re.search(r"(\d+)/(\d+)", cell)
            if not match or tuple(map(int, match.groups())) != expected:
                errors.append(f"{path.relative_to(REPO)}:{line_no}: {cell} != {expected}")
            percent = re.search(r"\((\d+(?:\.\d+)?)%\)", cell)
            if percent:
                digits = len(percent[1].partition(".")[2])
                value = 100 * expected[0] / expected[1]
                if percent[1] != f"{value:.{digits}f}":
                    errors.append(f"{path}:{line_no}: percentage differs")
            checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="optional audit evidence CSV")
    args = parser.parse_args()
    tracked = set(subprocess.check_output(
        ["git", "ls-files"], cwd=REPO, text=True).splitlines())
    hot = read(REPO / "data/targets/hotpotqa.json")
    mus = read(REPO / "research/frozen/musique_targets_59.json")
    errors, evidence, paper, transfers = [], [], {}, {}
    paths = sorted((REPO / "results").glob("*.json"))
    for path in paths:
        d = read(path)
        if not isinstance(d, dict) or not {"clean", "targets"} <= d.keys():
            continue
        targets, model = d["targets"], d["meta"]["model"]
        musique = "musique" in d["meta"]["kb_collection"]
        source = mus if musique else hot
        if set(targets) != set(source):
            errors.append(f"{path.name}: target question set differs")
        for q, t in targets.items():
            for key, alias in [("question", "question"), ("gold", "correct answer"),
                               ("wrong", "incorrect answer")]:
                if q not in source or t[key] != source[q].get(key, source[q].get(alias)):
                    errors.append(f"{path.name}: {q} {key} differs")
        clean = d["clean"]["answers"]
        current = read(main_path(model, musique))["clean"]["answers"]
        groups = d.get("variants", {"eval-after": {"after": d.get("after", {})}})
        for setting, r in groups.items():
            stored = r.get("after", {})
            if "answers" not in stored:
                continue
            after = stored["answers"]
            for name, records in [("clean", clean), ("after", after), ("main clean", current)]:
                if set(records) != set(targets):
                    errors.append(f"{path.name}: {name} question coverage differs")
            if any(set(a) != set(targets) for a in [clean, current, after]):
                continue
            own, canonical = score(targets, clean, after), score(targets, current, after)
            for key, value in [("flips_knowledge", own["flip"]),
                               ("flips_collapse", own["collapse"]),
                               ("flips", own["flip"] + own["collapse"]),
                               ("asr_unified", own["asr"])]:
                if key in stored and stored[key] != value:
                    errors.append(f"{path.name}: stored {key} does not match its own answers")
            cor = r.get("co_retrieval", {})
            if cor:
                if set(cor["per_target"]) != set(targets):
                    errors.append(f"{path.name}: retrieval coverage differs")
                true_count = sum(v["true_rank"] is not None for v in cor["per_target"].values())
                if true_count != cor["true_in_top8"]:
                    errors.append(f"{path.name}: retrieval total differs")
            writes = r.get("poison_writes", [])
            if writes:
                if len(writes) != r["n_poison"] or len({w["qid"] for w in writes}) != r["n_targets_fired"]:
                    errors.append(f"{path.name}: write totals differ")
            changed = sum(unified.correct(clean[q]["pred"], t["gold"])
                          != unified.correct(current[q]["pred"], t["gold"])
                          for q, t in targets.items())
            row = dict(file=str(path.relative_to(REPO)), dataset="musique" if musique else "hotpotqa",
                       model=model, method=method_name(path.name), setting=setting,
                       tracked=str(path.relative_to(REPO)) in tracked, **canonical,
                       stored_clean=own["clean"], stored_flip=own["flip"],
                       stored_collapse=own["collapse"], clean_membership_changes=changed,
                       written_questions=len({w["qid"] for w in writes}) if writes else "",
                       poison_chunks=len(writes) if writes else "",
                       true_in_top8=cor.get("true_in_top8", ""))
            evidence.append(row)
            if path == main_path(model, musique) or row["method"] in {"PR", "AP", "CR-AS", "CR-AK"}:
                paper[musique, model, row["method"]] = row
            if row["method"] == "transfer":
                attacker = path.name.removeprefix("08_longtail_xsm_").split("_to_")[0]
                transfers[attacker, model] = row
            if cor:
                summarized = summary.summarize(str(path), set(mus) if musique else None)
                sr = next(v for v in summarized if v["variant"] == setting)
                for key, alias in [("n", "n"), ("clean", "clean_em"), ("flip", "flips"),
                                   ("collapse", "flips_collapse"), ("asr", "asr")]:
                    if canonical[key] != sr[alias]:
                        errors.append(f"{path.name}: summary {key} differs")
    checked = sum(check_tables(REPO / f, paper, transfers, errors) for f in [
        "README.md", "baseline/poisonedrag/README.md", "baseline/agentpoison/README.md",
        "baseline/corruptrag/README.md", "results/cross_model_summary.md"])
    existing = (REPO / "results/ablation_summary.md").read_text()
    for build in [summary.build_main_table, summary.build_volume, summary.build_diversity,
                  summary.build_trigger, summary.build_topk, summary.build_cross_model,
                  summary.raw_all_runs]:
        if build() not in existing:
            errors.append(f"ablation_summary.md: {build.__name__} is out of date")
    # Reproduce what the published code can see, without touching live results.
    with tempfile.TemporaryDirectory(prefix="agenticrag_release_audit_") as folder:
        for path in paths:
            if str(path.relative_to(REPO)) in tracked:
                (Path(folder) / path.name).symlink_to(path)
        original = summary.RESULTS
        try:
            summary.RESULTS = folder
            missing_release_rows = sum(build().count("pending") for build in [
                summary.build_volume, summary.build_diversity,
                summary.build_trigger, summary.build_topk])
        finally:
            summary.RESULTS = original
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(evidence[0]))
            writer.writeheader()
            writer.writerows(evidence)
    print(f"Raw result JSONs: {len(paths)} local, "
          f"{sum(str(p.relative_to(REPO)) in tracked for p in paths)} tracked")
    print(f"Complete evaluations: {len(evidence)}; numeric table cells checked: {checked}")
    print(f"Target/answer coverage, stored arithmetic, paper numeric cells, summary errors: {len(errors)}")
    print(f"Clean snapshots different from current main table: "
          f"{sum(bool(r['clean_membership_changes']) for r in evidence)}")
    print(f"Evaluation files absent from Git release: {sum(not r['tracked'] for r in evidence)}")
    print(f"HotpotQA ablation rows become pending using only release files: {missing_release_rows}")
    print("Narrative claims and scoring limitations require the accompanying written audit.")
    for error in errors:
        print("ERROR:", error)
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
