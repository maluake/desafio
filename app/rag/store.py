"""Local persisted vector store + hybrid retriever.

Storage layout (``data/index/``)::

    manifest.json   backend, build time, sources, chunk count
    chunks.jsonl    one chunk per line (text + metadata)
    vectors.npy     L2-normalised embedding matrix (row i <-> chunk i)
    tfidf.pkl       fitted vectorizers (tfidf backend only)

Retrieval = dense/TF-IDF cosine + BM25, fused with Reciprocal Rank Fusion (RRF), then
per-document diversification so the top-k is not five chunks of the same page.
For production this interface maps directly onto pgvector / Qdrant / OpenSearch.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.core.observability import RAG_RETRIEVALS, get_logger, span
from app.rag.chunking import Chunk
from app.rag.embeddings import Embedder, load_embedder, make_embedder
from app.rag.text import expand_query, tokenize

log = get_logger(__name__)

RRF_K = 60


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity of the (expanded) query, 0..1 - used for thresholding
    rrf: float  # fused rank score - used for ordering


class VectorStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder) -> None:
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder
        self.bm25 = BM25Okapi([tokenize(c.text) or ["_"] for c in chunks]) if chunks else None

    # ----------------------------------------------------------------- build / io
    @classmethod
    def build(cls, chunks: list[Chunk], backend: str = "tfidf") -> VectorStore:
        embedder = make_embedder(backend)
        texts = [c.text for c in chunks]
        embedder.fit(texts)
        vectors = embedder.encode(texts) if texts else np.zeros((0, 1), dtype=np.float32)
        return cls(chunks, vectors, embedder)

    def save(self, path: Path, sources: list[str] | None = None) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(path / "vectors.npy", self.vectors)
        self.embedder.save(path)
        manifest = {
            "backend": self.embedder.name,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "chunks": len(self.chunks),
            "documents": len({c.doc_id for c in self.chunks}),
            "sources": sources or sorted({c.url for c in self.chunks if c.url}),
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> VectorStore:
        manifest = json.loads((path / "manifest.json").read_text())
        with open(path / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
        vectors = np.load(path / "vectors.npy")
        embedder = load_embedder(manifest["backend"], path)
        return cls(chunks, vectors, embedder)

    # ----------------------------------------------------------------- retrieval
    def search(self, query: str, top_k: int = 5, per_doc: int = 2) -> list[RetrievedChunk]:
        if not self.chunks:
            return []
        with span("rag.search", query=query, top_k=top_k) as s:
            expanded = expand_query(query)
            qvec = self.embedder.encode([expanded])[0]
            cos = self.vectors @ qvec
            dense_rank = np.argsort(-cos)

            bm25_scores = self.bm25.get_scores(tokenize(expanded)) if self.bm25 else np.zeros(len(self.chunks))
            sparse_rank = np.argsort(-bm25_scores)

            fused: dict[int, float] = {}
            for rank, idx in enumerate(dense_rank[:50]):
                fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank)
            for rank, idx in enumerate(sparse_rank[:50]):
                if bm25_scores[idx] <= 0:
                    break
                fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank)

            ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
            results: list[RetrievedChunk] = []
            per_doc_count: dict[str, int] = {}
            for idx, rrf in ordered:
                chunk = self.chunks[idx]
                if per_doc_count.get(chunk.doc_id, 0) >= per_doc:
                    continue
                per_doc_count[chunk.doc_id] = per_doc_count.get(chunk.doc_id, 0) + 1
                results.append(RetrievedChunk(chunk=chunk, score=float(cos[idx]), rrf=rrf))
                if len(results) >= top_k:
                    break
            s.attributes["hits"] = [r.chunk.chunk_id for r in results]
            s.attributes["top_score"] = round(results[0].score, 3) if results else 0.0
        RAG_RETRIEVALS.labels("hit" if results else "miss").inc()
        return results
