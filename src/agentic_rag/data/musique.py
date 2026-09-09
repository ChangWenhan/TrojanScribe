"""MuSiQue loader (dataset ablation).

Adapts musique_ans_v1.0_dev.jsonl (Trivedi et al., TACL 2022;
github.com/stonybrooknlp/musique) to the HotpotQuestion interface so the
whole 08_longtail / payload / victim pipeline runs unchanged.

Field mapping:
  qid               <- id
  answer            <- answer
  supporting_facts  <- [(title, idx) for paragraphs with is_supporting]
                       (idx plays the role of the HotpotQA sentence id)
  context           <- [(title, [paragraph_text]) for every paragraph]
  qtype             <- qtype (e.g. "bridge_2hop"), level = "ans"
"""
from __future__ import annotations

import json
import random

from .hotpot import HotpotQuestion
from ..kb.store import KnowledgeStore


def load_musique(path: str, n_questions: int = 2500, seed: int = 42) -> list[HotpotQuestion]:
    raw = [json.loads(line) for line in open(path) if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(raw)
    selected = raw[:n_questions]
    out = []
    for q in selected:
        supporting = [(p["title"], str(p["idx"])) for p in q["paragraphs"] if p["is_supporting"]]
        context = [(p["title"], [p["paragraph_text"]]) for p in q["paragraphs"]]
        out.append(
            HotpotQuestion(
                qid=q["id"],
                question=q["question"],
                answer=q["answer"],
                supporting_facts=supporting,
                context=context,
                qtype=q.get("qtype", ""),
                level="ans",
            )
        )
    return out


def build_musique_kb(store: KnowledgeStore, questions: list[HotpotQuestion],
                     author_id: str = "system") -> int:
    """Write all context paragraphs (deduped by title) with musique: provenance."""
    seen: set[str] = set()
    texts, authors, sources = [], [], []
    for q in questions:
        for title, text in q.paragraphs():
            if title in seen:
                continue
            seen.add(title)
            texts.append(text)
            authors.append(author_id)
            sources.append(f"musique:{q.qid}:{title}")
    store.write_many(texts, authors, sources, is_poisons=[False] * len(texts))
    return len(texts)
