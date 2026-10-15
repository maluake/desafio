"""Composition root: builds and wires every long-lived service once per process."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from app.config import Settings
from app.core.llm import LLMClient
from app.core.memory import ConversationMemory
from app.core.observability import get_logger
from app.data.repository import CustomerRepository, TicketStore
from app.rag.ingest import build_index
from app.rag.store import VectorStore
from app.tools.web import WebTools

log = get_logger(__name__)


@dataclass
class Services:
    settings: Settings
    llm: LLMClient | None
    kb: VectorStore
    customers: CustomerRepository
    tickets: TicketStore
    web: WebTools
    memory: ConversationMemory


def load_knowledge_base(settings: Settings) -> VectorStore:
    manifest = settings.index_dir / "manifest.json"
    if manifest.exists():
        try:
            store = VectorStore.load(settings.index_dir)
            log.info("knowledge base loaded", extra={"chunks": len(store.chunks)})
            return store
        except Exception as exc:  # corrupted / incompatible index: rebuild
            log.warning("could not load index, rebuilding", extra={"error": str(exc)})
    mode = "crawl" if settings.ingest_on_startup == "crawl" else "seed"
    return build_index(settings, mode)


def build_services(
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    web: WebTools | None = None,
    today: date | None = None,
) -> Services:
    if llm is None and settings.resolved_provider == "anthropic":
        llm = LLMClient(settings)
    log.info("llm provider", extra={"provider": "anthropic" if llm else "offline", "model": settings.llm_model})
    return Services(
        settings=settings,
        llm=llm,
        kb=load_knowledge_base(settings),
        customers=CustomerRepository(settings.customers_file, today=today),
        tickets=TicketStore(settings.runtime_dir),
        web=web or WebTools(settings),
        memory=ConversationMemory(settings.memory_max_turns, settings.memory_ttl_s),
    )
