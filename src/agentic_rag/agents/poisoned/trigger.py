"""Trigger designs for the poisoned subagent. Fired on the *task text* the
agent receives from its (victim) user."""
from __future__ import annotations

import random

import numpy as np

from ...kb.store import KnowledgeStore


class Trigger:
    name = "base"

    def check(self, task_text: str) -> bool:
        raise NotImplementedError


class KeywordTrigger(Trigger):
    """Fires when any trigger keyword appears in the task text."""

    name = "keyword"

    def __init__(self, keywords: list[str]):
        self.keywords = [k.lower() for k in keywords]

    def check(self, task_text: str) -> bool:
        t = task_text.lower()
        return any(k in t for k in self.keywords)


class SemanticTrigger(Trigger):
    """Fires when the task text is semantically similar (embedding cosine) to
    any of the attacker's target queries."""

    name = "semantic"

    def __init__(self, store: KnowledgeStore, target_queries: list[str], threshold: float = 0.82):
        self.store = store
        self.target_queries = target_queries
        self.threshold = threshold
        from ...kb.store import BGE_QUERY_PREFIX

        self._target_embs = np.array(
            store.embed([BGE_QUERY_PREFIX + q for q in target_queries])
        )

    def check(self, task_text: str) -> bool:
        from ...kb.store import BGE_QUERY_PREFIX

        emb = np.array(self.store.embed([BGE_QUERY_PREFIX + task_text])[0])
        sims = (self._target_embs @ emb) / (
            np.linalg.norm(self._target_embs, axis=1) * np.linalg.norm(emb) + 1e-9
        )
        return float(sims.max()) >= self.threshold


class AlwaysTrigger(Trigger):
    """Fires with probability p on every task (low-frequency persistent)."""

    name = "always"

    def __init__(self, p: float = 0.1):
        self.p = p

    def check(self, task_text: str) -> bool:
        return random.random() < self.p


def build_trigger(kind: str, store: KnowledgeStore, cfg: dict, target_queries: list[str]) -> Trigger:
    if kind == "keyword":
        keywords = cfg.get("trigger_keywords") or _auto_keywords(target_queries)
        return KeywordTrigger(keywords)
    if kind == "semantic":
        return SemanticTrigger(store, target_queries, cfg.get("semantic_threshold", 0.82))
    if kind == "always":
        return AlwaysTrigger(p=cfg.get("always_probability", 0.1))
    raise ValueError(f"unknown trigger kind: {kind}")


def _auto_keywords(target_queries: list[str]) -> list[str]:
    """Extract likely proper-noun keywords from target question texts."""
    stop = {"the", "a", "an", "of", "in", "on", "at", "was", "were", "is", "are", "did",
            "do", "and", "or", "to", "from", "with", "for", "that", "which", "who",
            "what", "when", "where", "how", "why", "before", "after", "his", "her",
            "their", "its", "this", "these", "those", "as", "by", "than"}
    kws: set[str] = set()
    for q in target_queries:
        for w in q.split():
            w0 = w.strip(",.()'\"")
            wl = w0.lower()
            if w0 and w0[0].isupper() and len(wl) > 3 and wl not in stop:
                kws.add(wl)
    return sorted(kws)