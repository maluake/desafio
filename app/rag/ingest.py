"""Ingestion pipeline: load -> clean -> chunk -> embed -> persist.

Two sources are combined:

1. **Live crawl** of the Getnet websites (``CRAWL_URLS``): polite BFS crawler that honours
   robots.txt, stays on the seed domains, strips navigation/boilerplate and keeps the page
   ``<title>``/``<h1-h3>`` structure as Markdown headings so the chunker can split by section.
2. **Curated seed corpus** (``data/seed/*.md``): hand-written, source-attributed notes that
   cover the challenge scenarios. It guarantees a working knowledge base when the site is
   unreachable (CI, air-gapped demos, bot protection) and acts as a "golden" layer that
   product/CX teams can edit without touching code.

Usage::

    python -m app.rag.ingest --mode seed     # seed corpus only (default in Docker build)
    python -m app.rag.ingest --mode crawl    # crawl + seed
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
import yaml
from bs4 import BeautifulSoup

from app.config import Settings, get_settings
from app.core.observability import configure_logging, get_logger
from app.rag.chunking import Chunk, Document, chunk_document
from app.rag.store import VectorStore

log = get_logger(__name__)

USER_AGENT = "GetnetSupportBot/1.0 (+https://github.com/maluake/desafio; RAG ingestion)"
_SKIP_EXT = re.compile(r"\.(png|jpe?g|gif|svg|webp|ico|css|js|zip|mp4|woff2?|ttf)$", re.I)


# --------------------------------------------------------------------------- seed


def load_seed_documents(seed_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(seed_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta: dict = {}
        body = raw
        if raw.startswith("---"):
            _, fm, body = raw.split("---", 2)
            meta = yaml.safe_load(fm) or {}
        docs.append(
            Document(
                doc_id=f"seed:{path.stem}",
                title=meta.get("title", path.stem.replace("-", " ").title()),
                text=body.strip(),
                url=meta.get("url"),
                lang=meta.get("lang", "en"),
                metadata={"source_type": "curated", "references": meta.get("references", [])},
            )
        )
    log.info("loaded seed documents", extra={"count": len(docs)})
    return docs


# -------------------------------------------------------------------------- crawl


def html_to_markdown(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "form", "header", "footer", "nav", "aside"]):
        tag.decompose()
    for tag in soup.select('[class*="cookie"], [id*="cookie"], [class*="menu"], [aria-hidden="true"]'):
        tag.decompose()
    title = (soup.title.string or "").strip() if soup.title else ""
    main = soup.find("main") or soup.body or soup
    lines: list[str] = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"]):
        text = " ".join(el.get_text(" ", strip=True).split())
        if len(text) < 3:
            continue
        if el.name in {"h1", "h2", "h3", "h4"}:
            lines.append(f"\n{'#' * int(el.name[1])} {text}\n")
        elif el.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)
    md = "\n".join(lines)
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return title, md


def crawl(seeds: list[str], max_pages: int, timeout: float = 15.0, delay_s: float = 0.5) -> list[Document]:
    allowed_hosts = {urlparse(u).netloc for u in seeds}
    robots: dict[str, RobotFileParser] = {}
    queue: deque[str] = deque(seeds)
    seen: set[str] = set()
    docs: list[Document] = []
    seen_hashes: set[str] = set()

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=timeout, follow_redirects=True) as client:
        while queue and len(docs) < max_pages:
            url = urldefrag(queue.popleft())[0]
            if url in seen or _SKIP_EXT.search(url):
                continue
            seen.add(url)
            host = urlparse(url).netloc
            if host not in robots:
                rp = RobotFileParser()
                try:
                    r = client.get(f"{urlparse(url).scheme}://{host}/robots.txt")
                    rp.parse(r.text.splitlines() if r.status_code == 200 else [])
                except httpx.HTTPError:
                    rp.parse([])
                robots[host] = rp
            if not robots[host].can_fetch(USER_AGENT, url):
                continue
            try:
                resp = client.get(url)
            except httpx.HTTPError as exc:
                log.warning("crawl failed", extra={"url": url, "error": str(exc)})
                continue
            if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
                continue
            title, md = html_to_markdown(resp.text)
            digest = hashlib.sha1(md.encode(), usedforsecurity=False).hexdigest()
            if len(md) > 200 and digest not in seen_hashes:
                seen_hashes.add(digest)
                lang = "en" if "/en" in urlparse(url).path else "pt"
                docs.append(
                    Document(
                        doc_id=f"web:{hashlib.sha1(url.encode(), usedforsecurity=False).hexdigest()[:10]}",
                        title=title or url,
                        text=f"# {title}\n\n{md}" if title else md,
                        url=url,
                        lang=lang,
                        metadata={"source_type": "crawl"},
                    )
                )
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                nxt = urldefrag(urljoin(url, a["href"]))[0]
                if urlparse(nxt).netloc in allowed_hosts and nxt not in seen:
                    queue.append(nxt)
            time.sleep(delay_s)
    log.info("crawl finished", extra={"pages": len(docs), "visited": len(seen)})
    return docs


# ------------------------------------------------------------------------- build


def build_index(settings: Settings, mode: str = "seed") -> VectorStore:
    docs = load_seed_documents(settings.seed_dir)
    if mode == "crawl":
        try:
            docs = crawl(settings.crawl_urls, settings.crawl_max_pages) + docs
        except Exception as exc:  # crawl is best-effort; the seed corpus keeps the service usable
            log.warning("crawl aborted, continuing with seed corpus", extra={"error": str(exc)})
    chunks: list[Chunk] = []
    for d in docs:
        chunks.extend(chunk_document(d, settings.chunk_size, settings.chunk_overlap))
    store = VectorStore.build(chunks, settings.embeddings_backend)
    store.save(settings.index_dir)
    log.info("index built", extra={"documents": len(docs), "chunks": len(chunks), "path": str(settings.index_dir)})
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Getnet knowledge base index")
    parser.add_argument("--mode", choices=["seed", "crawl"], default="seed")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    build_index(settings, args.mode)


if __name__ == "__main__":
    main()
