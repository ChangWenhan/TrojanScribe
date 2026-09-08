"""Shared helpers for experiment scripts: config, store, question sets, results."""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
RESULTS = os.path.join(REPO, "results")


def load_config(path: str | None = None) -> dict:
    path = path or os.path.join(REPO, "configs", "default.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Remote-inference mode: AGENTIC_RAG_BASE_URL points the local-vLLM client
    # at another machine's OpenAI-compatible API (the 141 node serves models,
    # this node runs the experiment code + KB only).
    base_url = os.environ.get("AGENTIC_RAG_BASE_URL")
    if base_url:
        cfg["llm"]["local"]["base_url"] = base_url
    return cfg


def get_store(config: dict) -> object:
    from agentic_rag.kb.store import KnowledgeStore

    kb = config["kb"]
    emb = config["embedding"]
    store = KnowledgeStore(
        persist_dir=os.path.join(REPO, kb["persist_dir"]),
        collection=kb["collection"],
        model_path=emb["model_path"],
        dim=emb["dim"],
    )
    return store


def get_questions(config: dict) -> list:
    from agentic_rag.data.hotpot import load_hotpot

    corpus = config["corpus"]
    return load_hotpot(
        os.path.join(REPO, corpus["path"]),
        n_questions=corpus["n_questions"],
        seed=corpus["seed"],
    )


def ensure_kb(config: dict, questions: list, force: bool = False) -> None:
    from agentic_rag.data.hotpot import build_kb

    store = get_store(config)
    if force:
        store.reset()
    if force or store.count(poison_only=False) == 0:
        n = build_kb(store, questions)
        print(f"[kb] built from {len(questions)} questions -> {n} chunks")
    else:
        print(f"[kb] existing: {store.count()} chunks (system={store.count(False)-store.count(True)} clean)")


def make_llm(config: dict, role: str = "agent"):
    from agentic_rag.llm import LLMBackend, load_llm_config

    return LLMBackend(load_llm_config(config, role))


def split_sets(questions: list, n_targets: int, seed: int = 7) -> tuple[list, list]:
    """Return (target_questions, control_questions), disjoint."""
    from agentic_rag.data.hotpot import pick_entity_questions

    targets = pick_entity_questions(questions, n_targets, seed=seed)
    target_ids = {t.qid for t in targets}
    controls = [q for q in questions if q.qid not in target_ids][:n_targets]
    return targets, controls


_AUTO_RUN_STAMP: str | None = None  # per-process cache for the auto run id


def save_results(name: str, data: dict) -> str:
    """Persist one result. Each *run* gets its own timestamped directory so
    runs are distinguishable and never silently overwritten:

      results/runs/<run_id>/<name>.json   the actual result of this run

    The root file results/<name>.json is kept as a *mirror* of the latest run
    (the same data), because downstream consumers (summarize_ablation.py,
    13_unified_eval.py) and the run scripts read fixed result names.

    run_id comes from AGENTIC_RAG_RUN_ID when set (run scripts set it, e.g.
    "vol2_20260907_103000", and put the run log in the same directory); if it
    is unset, a per-process auto id "<name>_<timestamp>" is used so repeated
    incremental saves inside one process share the same directory.
    """
    os.makedirs(RESULTS, exist_ok=True)
    run_id = os.environ.get("AGENTIC_RAG_RUN_ID", "").strip()
    if run_id:
        run_dir = os.path.join(RESULTS, "runs", run_id)
    else:
        global _AUTO_RUN_STAMP
        if _AUTO_RUN_STAMP is None:
            _AUTO_RUN_STAMP = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(RESULTS, "runs", f"{name}_{_AUTO_RUN_STAMP}")
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    # root mirror = latest run, for fixed-name consumers
    mirror = os.path.join(RESULTS, f"{name}.json")
    with open(mirror, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[results] saved -> {path}\n[results] mirror -> {mirror}")
    return path


def save_csv(name: str, rows: list[dict]) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{name}.csv")
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"[results] saved -> {path}")
    return path


def answer_questions(agent, questions: list, label: str) -> dict:
    """Run victim agent over questions; return per-qid results."""
    out = {}
    for i, q in enumerate(questions):
        res = agent.ask(q.question)
        pred = res["answer"]
        out[q.qid] = {
            "question": q.question,
            "gold": q.answer,
            "pred": pred,
            "correct": _correct(pred, q.answer),
            "n_tool_calls": res["n_tool_calls"],
            "trace": res["trace"],
        }
        if (i + 1) % 10 == 0 or i == len(questions) - 1:
            print(f"  [{label}] {i+1}/{len(questions)} done")
    return out


def _correct(pred: str, gold: str) -> bool:
    from agentic_rag.eval.metrics import exact_match

    return exact_match(pred, gold)


def make_victim(store, config, top_k: int | None = None, system_prompt: str | None = None):
    from agentic_rag.agents.base import SYSTEM_VICTIM, ReActAgent, make_kb_search_tool
    from agentic_rag.llm import load_llm_config

    llm_cfg = load_llm_config(config, "agent")
    tools = [make_kb_search_tool(store, top_k or config["kb"]["top_k"])]
    return ReActAgent(
        name="victim",
        llm_cfg=llm_cfg,
        store=store,
        system_prompt=system_prompt or SYSTEM_VICTIM,
        tools=tools,
        top_k=top_k or config["kb"]["top_k"],
        max_iterations=config["attack"].get("max_iterations", 8),
    )