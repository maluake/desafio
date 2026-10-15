"""Structure-aware chunking.

Documents are split on Markdown headings first (so a chunk never mixes two products),
then packed paragraph-by-paragraph up to ``chunk_size`` characters with ``overlap``
characters of trailing context. Every chunk is prefixed with its heading path
("Card machines > Get Smart") which gives both retrievers and the LLM local context.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

_HEADING = re.compile(r"^(#{1,4})\s+(.*)$")


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    url: str | None = None
    lang: str = "en"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    url: str | None = None
    lang: str = "en"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        return cls(**d)


def _sections(text: str, title: str) -> list[tuple[str, str]]:
    path: list[str] = []
    sections: list[tuple[str, str]] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(path) or title, body))
        buf.clear()

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    flush()
    return sections


def _tail(text: str, overlap: int) -> str:
    """Last ``overlap`` characters, starting at a word boundary (never mid-word)."""
    if not overlap or len(text) <= overlap:
        return text if overlap else ""
    tail = text[-overlap:]
    cut = tail.find(" ")
    return tail[cut + 1 :] if cut != -1 else tail


def _pack(paragraphs: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        is_table = all(line.lstrip().startswith("|") for line in para.splitlines())
        if is_table and len(para) <= 2 * size:  # keep tables atomic: a half table is useless
            if current.strip():
                chunks.append(current.strip())
            chunks.append(para)
            current = ""
            continue
        if len(para) > size:  # long paragraph / list: split on lines, then on sentences
            units = [u for line in para.splitlines() for u in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line) if u.strip()]
            for unit in units:
                if len(current) + len(unit) + 1 > size and current:
                    chunks.append(current.strip())
                    current = _tail(current, overlap)
                current += "\n" + unit
            continue
        if len(current) + len(para) + 2 > size and current:
            chunks.append(current.strip())
            current = _tail(current, overlap)
        current += "\n\n" + para
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_document(doc: Document, size: int = 900, overlap: int = 150) -> list[Chunk]:
    out: list[Chunk] = []
    for section, body in _sections(doc.text, doc.title):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        for i, piece in enumerate(_pack(paragraphs, size, overlap)):
            digest = hashlib.sha1(f"{doc.doc_id}|{section}|{i}|{piece}".encode(), usedforsecurity=False).hexdigest()[:12]
            out.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{digest}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    section=section,
                    text=f"{section}\n{piece}",
                    url=doc.url,
                    lang=doc.lang,
                    metadata=doc.metadata,
                )
            )
    return out
