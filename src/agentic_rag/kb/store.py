"""Shared knowledge store backed by ChromaDB.

Every chunk carries provenance metadata:
    author_id, timestamp, source, is_poison
The `is_poison` flag is for evaluation bookkeeping only (the store API
mirrors what a real production KB would expose; flags are inspected by
our metrics, not by agents).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class BgeEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_path: str, dim: int = 768):
        # Keep embeddings on CPU so they never contend with the serving GPU (vLLM).
        self._model = SentenceTransformer(model_path, device="cpu")
        self._dim = dim

    def __call__(self, input: Documents) -> Embeddings:
        emb = self._model.encode(
            list(input), normalize_embeddings=True, show_progress_bar=False
        )
        return [e.tolist() for e in emb]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self(list(texts))


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    author_id: str
    source: str
    is_poison: bool
    timestamp: float
    meta: dict = field(default_factory=dict)

    @property
    def provenance(self) -> dict:
        return dict(author_id=self.author_id, source=self.source, is_poison=self.is_poison)


class KnowledgeStore:
    def __init__(self, persist_dir: str, collection: str, model_path: str, dim: int = 768):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._ef = BgeEmbeddingFunction(model_path, dim=dim)
        self._coll = self._client.get_or_create_collection(
            name=collection, embedding_function=self._ef, metadata={"hnsw:space": "cosine"}
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._ef.embed(texts)

    def write_many(
        self,
        texts: list[str],
        author_ids: list[str],
        sources: list[str],
        is_poisons: list[bool] | None = None,
        batch: int = 512,
    ) -> list[str]:
        """Batch write (avoids per-chunk commit overhead)."""
        ids = [uuid.uuid4().hex[:16] for _ in texts]
        now = time.time()
        metas = [
            {
                "author_id": author_ids[i],
                "source": sources[i],
                "is_poison": int(is_poisons[i]) if is_poisons else 0,
                "timestamp": now,
            }
            for i in range(len(texts))
        ]
        for b in range(0, len(texts), batch):
            self._coll.add(
                ids=ids[b : b + batch],
                documents=texts[b : b + batch],
                metadatas=metas[b : b + batch],
            )
        return ids

    def write(
        self,
        text: str,
        author_id: str,
        source: str,
        is_poison: bool = False,
        timestamp: float | None = None,
        extra: dict | None = None,
    ) -> str:
        cid = uuid.uuid4().hex[:16]
        meta = {
            "author_id": author_id,
            "source": source,
            "is_poison": int(is_poison),
            "timestamp": timestamp if timestamp is not None else time.time(),
        }
        if extra:
            meta.update(extra)
        self._coll.add(ids=[cid], documents=[text], metadatas=[meta])
        return cid

    def search(
        self, query: str, k: int = 5, author_filter: str | None = None
    ) -> list[RetrievedChunk]:
        # bge-base-en-v1.5: queries need the instruction prefix (documents do not).
        query_text = BGE_QUERY_PREFIX + query
        where = {"author_id": author_filter} if author_filter else None
        res = self._coll.query(
            query_texts=[query_text], n_results=k, where=where, include=["documents", "metadatas", "distances"]
        )
        chunks: list[RetrievedChunk] = []
        for i, doc in enumerate(res["documents"][0]):
            m = res["metadatas"][0][i]
            chunks.append(
                RetrievedChunk(
                    id=res["ids"][0][i],
                    text=doc,
                    score=1.0 - res["distances"][0][i],
                    author_id=m["author_id"],
                    source=m["source"],
                    is_poison=bool(m["is_poison"]),
                    timestamp=m["timestamp"],
                    meta=m,
                )
            )
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks

    def count(self, poison_only: bool | None = None) -> int:
        if poison_only is None:
            return self._coll.count()
        res = self._coll.get(where={"is_poison": int(poison_only)})
        return len(res["ids"])

    def all_chunks(self) -> list[RetrievedChunk]:
        res = self._coll.get(include=["documents", "metadatas"])
        chunks = []
        for i, doc in enumerate(res["documents"]):
            m = res["metadatas"][i]
            chunks.append(
                RetrievedChunk(
                    id=res["ids"][i],
                    text=doc,
                    score=0.0,
                    author_id=m["author_id"],
                    source=m["source"],
                    is_poison=bool(m["is_poison"]),
                    timestamp=m["timestamp"],
                    meta=m,
                )
            )
        return chunks

    def delete(self, ids: list[str]) -> None:
        self._coll.delete(ids=ids)

    def delete_poison(self) -> None:
        res = self._coll.get(where={"is_poison": 1})
        if res["ids"]:
            self._coll.delete(ids=res["ids"])

    def delete_writer(self, author_id: str) -> int:
        """Remove all chunks written by a given author (cleanup for re-runs).
        Returns the number of deleted chunks."""
        res = self._coll.get(where={"author_id": author_id})
        if res["ids"]:
            self._coll.delete(ids=res["ids"])
        return len(res["ids"])

    def reset(self) -> None:
        self._client.delete_collection(self._coll.name)
        self._coll = self._client.get_or_create_collection(
            name=self._coll.name, embedding_function=self._ef, metadata={"hnsw:space": "cosine"}
        )

    def delete_collection(self) -> None:
        try:
            self._client.delete_collection(self._coll.name)
        except Exception:
            pass

    @classmethod
    def ephemeral(cls, persist_dir: str, collection: str, model_path: str, dim: int = 768) -> "KnowledgeStore":
        """Fresh store whose collection is wiped on construction (for ablations)."""
        s = cls(persist_dir, collection, model_path, dim)
        s.delete_collection()
        s = cls(persist_dir, collection, model_path, dim)
        return s