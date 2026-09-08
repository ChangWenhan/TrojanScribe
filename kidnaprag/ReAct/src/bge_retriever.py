"""BGE retriever adapter: exposes our KnowledgeStore (bge-base + chroma) with
the same interface as KidnapRAG's E5_Retriever (search -> list of dicts with
is_poisoned flags), so the ReAct agent runs against our KB with our retriever
(controlled-variable comparison: only the poison-doc construction differs)."""

import sys
import os

_PROJECT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, os.path.join(_PROJECT, "src"))

from agentic_rag.kb.store import KnowledgeStore


class BgeRetriever:
    """Adapter: KnowledgeStore (bge-base-en-v1.5, chroma, 65k HotpotQA chunks)."""

    def __init__(self, store: KnowledgeStore, model_name: str = "bge-base-en-v1.5",
                 device: str = "cpu", batch_size: int = 32, **kwargs):
        self.store = store
        self.num_clean_docs = store.count(poison_only=False)

    def search(self, query: str, k: int = 8):
        chunks = self.store.search(query, k=k)
        results = []
        for c in chunks:
            results.append({
                "id": c.id,
                "title": (c.meta or {}).get("title", "") or c.id,
                "contents": c.text,
                "score": float(c.score),
                "is_poisoned": bool(c.is_poison),
            })
        return results