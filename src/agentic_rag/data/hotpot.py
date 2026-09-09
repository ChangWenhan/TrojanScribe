"""HotpotQA loader: subset selection + KB construction from context paragraphs."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from ..kb.store import KnowledgeStore


@dataclass
class HotpotQuestion:
    qid: str
    question: str
    answer: str
    supporting_facts: list
    context: list  # list of (title, [sentences])
    qtype: str
    level: str

    def paragraphs(self) -> list[tuple[str, str]]:
        out = []
        for title, sents in self.context:
            out.append((title, " ".join(sents)))
        return out

    def supporting_titles(self) -> list[str]:
        return [sf[0] for sf in (self.supporting_facts or [])]

    def true_paragraph_text(self) -> str | None:
        """Text of a gold supporting paragraph (NOT the title).

        paragraphs()[0][0] is a title string; attack payloads that rewrite the
        'true paragraph' (AR3 bio, AR5 entity-swap) must get the actual text.
        Prefers the supporting paragraph that states the gold answer; falls
        back to the first supporting paragraph in context order.
        """
        titles = set(self.supporting_titles())
        if not titles:
            return None
        candidates = [
            (title, " ".join(sents))
            for title, sents in self.context
            if title in titles
        ]
        if not candidates:
            return None
        gold = self.answer.strip().lower()
        for title, text in candidates:
            if gold and gold in text.lower():
                return text
        return candidates[0][1]


def load_hotpot(path: str, n_questions: int = 500, seed: int = 42) -> list[HotpotQuestion]:
    raw = json.load(open(path))
    rng = random.Random(seed)
    rng.shuffle(raw)
    selected = raw[:n_questions]
    return [
        HotpotQuestion(
            qid=q["_id"],
            question=q["question"],
            answer=q["answer"],
            supporting_facts=q["supporting_facts"],
            context=q["context"],
            qtype=q["type"],
            level=q.get("level", ""),
        )
        for q in selected
    ]


def pick_entity_questions(questions: list[HotpotQuestion], n: int, seed: int = 42) -> list[HotpotQuestion]:
    """Pick questions whose gold answer is a proper noun (not yes/no), so poisoned
    passages can support a *different* concrete entity and be judged by EM/F1."""
    cands = [
        q
        for q in questions
        if q.answer.strip().lower() not in ("yes", "no", "unknown")
        and len(q.answer.strip()) > 1
        and (
            " " not in q.answer.strip().lower() or len(q.answer.strip()) >= 4
        )
    ]
    rng = random.Random(seed)
    rng.shuffle(cands)
    return cands[:n]


def build_kb(store: KnowledgeStore, questions: list[HotpotQuestion], author_id: str = "system") -> int:
    """Write all context paragraphs (deduped by title) as chunks with provenance."""
    seen: set[str] = set()
    texts, authors, sources = [], [], []
    for q in questions:
        for title, text in q.paragraphs():
            if title in seen:
                continue
            seen.add(title)
            texts.append(text)
            authors.append(author_id)
            sources.append(f"hotpotqa:{q.qid}:{title}")
    store.write_many(texts, authors, sources, is_poisons=[False] * len(texts))
    return len(texts)