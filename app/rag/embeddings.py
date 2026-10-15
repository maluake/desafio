"""Embedding backends.

* ``tfidf`` (default): word 1-2-gram + char 3-5-gram TF-IDF, L2-normalised. Zero downloads,
  deterministic, fast, and strong on short product-FAQ corpora with many proper nouns
  ("Get Smart", "Pix", "D+1"). Char n-grams make it robust to typos and PT inflections.
* ``fastembed``: multilingual dense embeddings (ONNX, CPU) for semantic recall across
  languages. Optional dependency: ``pip install fastembed``.

Both expose ``fit(texts)`` / ``encode(texts) -> np.ndarray`` (rows L2-normalised), so the
vector store is backend-agnostic.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from app.rag.text import normalize


class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> None: ...
    def encode(self, texts: list[str]) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self) -> None:
        self.word = TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.char = TfidfVectorizer(
            preprocessor=normalize, analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True
        )

    def fit(self, texts: list[str]) -> None:
        self.word.fit(texts)
        self.char.fit(texts)

    def encode(self, texts: list[str]) -> np.ndarray:
        w = self.word.transform(texts)
        c = self.char.transform(texts)
        m = sparse.hstack([w, c]).tocsr().astype(np.float32)
        dense = m.toarray()
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms

    def save(self, path: Path) -> None:
        with open(path / "tfidf.pkl", "wb") as f:
            pickle.dump({"word": self.word, "char": self.char}, f)

    @classmethod
    def load(cls, path: Path) -> TfidfEmbedder:
        obj = cls()
        with open(path / "tfidf.pkl", "rb") as f:  # our own artefact, produced by ingest
            data = pickle.load(f)  # noqa: S301
        obj.word, obj.char = data["word"], data["char"]
        return obj


class FastEmbedEmbedder:  # pragma: no cover - optional heavy dependency
    name = "fastembed"
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self) -> None:
        from fastembed import TextEmbedding

        self.model = TextEmbedding(self.model_name)

    def fit(self, texts: list[str]) -> None:
        return None

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self.model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def save(self, path: Path) -> None:
        (path / "fastembed.txt").write_text(self.model_name)

    @classmethod
    def load(cls, path: Path) -> FastEmbedEmbedder:
        return cls()


def make_embedder(backend: str) -> Embedder:
    return FastEmbedEmbedder() if backend == "fastembed" else TfidfEmbedder()


def load_embedder(backend: str, path: Path) -> Embedder:
    return FastEmbedEmbedder.load(path) if backend == "fastembed" else TfidfEmbedder.load(path)
