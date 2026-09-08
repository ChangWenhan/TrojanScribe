"""Build the MuSiQue KB (dataset ablation): embed all dev distractor paragraphs
into data/chroma_musique (collection musique_kb) with the same bge encoder.

Run once; idempotent (skips when the collection already has chunks).
CPU-only — safe to run next to the vLLM victim server.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from common import REPO, load_config  # noqa: E402

from agentic_rag.data.musique import build_musique_kb, load_musique  # noqa: E402
from agentic_rag.kb.store import KnowledgeStore  # noqa: E402


def main() -> None:
    config = load_config()
    store = KnowledgeStore(
        persist_dir=os.path.join(REPO, "data", "chroma_musique"),
        collection="musique_kb",
        model_path=config["embedding"]["model_path"],
        dim=config["embedding"]["dim"],
    )
    n = store.count()
    if n > 0:
        print(f"[musique-kb] existing collection has {n} chunks — nothing to do")
        return
    questions = load_musique(
        os.path.join(REPO, "data", "musique_ans_v1.0_dev.jsonl"),
        n_questions=2500, seed=42,
    )
    print(f"[musique-kb] building from {len(questions)} dev questions...")
    built = build_musique_kb(store, questions)
    print(f"[musique-kb] wrote {built} chunks -> data/chroma_musique (total {store.count()})")


if __name__ == "__main__":
    main()
