"""Experiment 14: KidnapRAG React baselines (controlled re-run).

Runs the four KidnapRAG official attacks on THEIR OWN ReAct agent inside the
kidnaprag repo, using their official poison generators' output
(results/poison_<method>.jsonl) against the shared bge KB (66,581 clean chunks,
top-8) and the shared 60 long-tail targets (kidnaprag/.../hotpotqa.json).

V2 (2026-09-08): victim model = GLM-4-9B (Qwen3-4B retired); the clean ReAct
baseline is re-run here too (it used to be a one-off historical file), and the
result files are named hotpotqa_seed1_<method>_xlam28b.json (suffix derived from
the served model name in attack_react.py).

Controlled variables (identical to experiment 08/langgraph side):
  victim model  xLAM-2-8B via vLLM (localhost:8000)
  KB            data/chroma (hotpot_kb) bge-base-en-v1.5
  targets       the same 60 qids used by the langgraph run
  poison docs   official KidnapRAG generators (same files as before)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KD = os.path.join(ROOT, "kidnaprag", "ReAct", "ReAct")
ADV_DIR = os.path.join(ROOT, "kidnaprag", "ReAct", "results", "adv_targeted_results")
METHODS = ["clean", "naive", "poisonedRAG", "ours", "topicattack"]
MODEL_NAME = "xlam-2-8b"
M_SUFFIX = re.sub(r"[^a-zA-Z0-9]", "", MODEL_NAME)

sys.path.insert(0, os.path.join(ROOT, "src"))


def inject(store, method: str, adv: dict) -> int:
    path = os.path.join(ROOT, "results", f"poison_{method}.jsonl")
    n = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        text = o.get("text", "")
        if not text:
            continue
        title = o.get("title", "")
        if title and not text.startswith(title):
            text = f"{title}\n{text}"
        meta = o.get("metadata") or {}
        qid = meta.get("original_id", "")
        store.write(
            text, author_id=f"attacker-{method}", source=f"kidnaprag-{method}",
            is_poison=True,
            extra={"title": title,
                   "target_question": adv.get(qid, {}).get("question", ""),
                   "wrong_answer": meta.get("target_answer", "")},
        )
        n += 1
    return n


def run_attack(method: str) -> None:
    env = dict(os.environ)
    env["OPENAI_BASE_URL"] = "http://localhost:8000/v1"
    env["OPENAI_API_KEY"] = "EMPTY"
    log = open(os.path.join(ROOT, "results", f"react_{method}.log"), "w")
    cmd = [
        sys.executable, "attack_react.py",
        "--dataset", "hotpotqa", "--seed", "1",
        "--model_path", MODEL_NAME, "--attack_method", method,
    ]
    proc = subprocess.run(cmd, cwd=KD, env=env, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    if proc.returncode != 0:
        raise RuntimeError(f"attack failed for {method} (rc={proc.returncode})")


def check_result(method: str) -> dict:
    path = os.path.join(ADV_DIR, f"hotpotqa_seed1_{method}_{M_SUFFIX}.json")
    if not os.path.exists(path):
        raise RuntimeError(f"expected result file missing: {path}")
    d = json.load(open(path))
    det = d["details"]
    n_err = sum(1 for x in det if x.get("error"))
    print(f"[14] {method}: n={len(det)} errors={n_err} "
          f"em={sum(1 for x in det if x['accuracy_em'])} "
          f"asr={sum(1 for x in det if x['asr_success'])}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(METHODS))
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    import yaml
    from agentic_rag.kb.store import KnowledgeStore

    config = yaml.safe_load(open(os.path.join(ROOT, "configs", "default.yaml")))
    store = KnowledgeStore(
        persist_dir=os.path.join(ROOT, "data", "chroma"),
        collection="hotpot_kb",
        model_path=config["embedding"]["model_path"],
        dim=config["embedding"]["dim"],
    )
    adv = json.load(open(os.path.join(ADV_DIR, "hotpotqa.json")))

    for method in methods:
        if method != "clean":
            # poison methods inject their corpus first; clean runs the pristine KB
            store.delete_poison()
            n = inject(store, method, adv)
            print(f"[14] {method}: injected {n} poison chunks", flush=True)
        run_attack(method)
        check_result(method)


if __name__ == "__main__":
    main()