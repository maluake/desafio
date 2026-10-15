"""Centralised, environment-driven configuration (12-factor style)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM -----------------------------------------------------------------
    # "auto" uses Anthropic when credentials are present, otherwise the deterministic
    # offline policies. "offline" forces the offline policies (tests / demos without a key).
    llm_provider: Literal["auto", "anthropic", "offline"] = "auto"
    anthropic_api_key: str | None = None
    llm_model: str = "claude-opus-5-5"
    router_model: str = "claude-opus-5-5"
    router_effort: Literal["low", "medium", "high"] = "low"
    agent_effort: Literal["low", "medium", "high"] = "medium"
    llm_max_tokens: int = 8000
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2
    llm_refusal_fallback: bool = True
    agent_max_tool_iterations: int = 6

    # --- RAG -----------------------------------------------------------------
    seed_dir: Path = ROOT_DIR / "data" / "seed"
    index_dir: Path = ROOT_DIR / "data" / "index"
    crawl_urls: list[str] = Field(
        default_factory=lambda: [
            "https://www.getnet.net/en",
            "https://www.getnet.net/",
            "https://site.getnet.com.br/",
        ]
    )
    crawl_max_pages: int = 60
    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.05
    embeddings_backend: Literal["tfidf", "fastembed"] = "tfidf"
    ingest_on_startup: Literal["none", "seed", "crawl"] = "seed"

    # --- External tools --------------------------------------------------------
    web_search_provider: Literal["ddgs", "tavily", "none"] = "ddgs"
    tavily_api_key: str | None = None
    http_timeout_s: float = 10.0

    # --- Customer data -----------------------------------------------------------
    customers_file: Path = ROOT_DIR / "data" / "customers" / "customers.json"
    runtime_dir: Path = ROOT_DIR / "data" / "runtime"

    # --- Orchestration / guardrails ----------------------------------------------
    router_confidence_threshold: float = 0.55
    max_message_chars: int = 2000
    memory_max_turns: int = 6
    memory_ttl_s: int = 1800

    # --- Human handoff ------------------------------------------------------------
    escalation_webhook_url: str | None = None  # Slack / Teams incoming webhook

    # --- Observability ------------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True
    expose_trace: bool = True

    @property
    def resolved_provider(self) -> Literal["anthropic", "offline"]:
        if self.llm_provider == "offline":
            return "offline"
        if self.llm_provider == "anthropic":
            return "anthropic"
        return "anthropic" if self.anthropic_api_key else "offline"


@lru_cache
def get_settings() -> Settings:
    return Settings()
