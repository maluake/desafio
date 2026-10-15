from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agents.orchestrator import Orchestrator
from app.config import ROOT_DIR, Settings
from app.container import Services, build_services
from app.core.llm import LLMClient
from app.main import create_app
from app.rag.ingest import build_index
from app.rag.store import VectorStore
from tests.fakes import FakeWebTools

# Thursday. "Yesterday" = Wednesday 2026-10-14, so D+1 debit lands on Thursday 15th.
TODAY = date(2026, 10, 15)


@pytest.fixture(scope="session")
def index_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("index")
    build_index(Settings(index_dir=path, seed_dir=ROOT_DIR / "data" / "seed", llm_provider="offline"), "seed")
    return path


@pytest.fixture(scope="session")
def kb(index_dir: Path) -> VectorStore:
    return VectorStore.load(index_dir)


@pytest.fixture
def settings(tmp_path: Path, index_dir: Path) -> Settings:
    return Settings(
        llm_provider="offline",
        index_dir=index_dir,
        runtime_dir=tmp_path / "runtime",
        log_json=False,
        expose_trace=True,
        escalation_webhook_url=None,
    )


@pytest.fixture
def web() -> FakeWebTools:
    return FakeWebTools()


@pytest.fixture
def services(settings: Settings, web: FakeWebTools) -> Services:
    return build_services(settings, web=web, today=TODAY)  # type: ignore[arg-type]


@pytest.fixture
def orchestrator(services: Services) -> Orchestrator:
    return Orchestrator(services)


@pytest.fixture
def client(services: Services) -> TestClient:
    return TestClient(create_app(services))


def make_llm_services(settings: Settings, messages_api, web=None) -> Services:  # noqa: ANN001
    llm = LLMClient(settings, messages_api=messages_api)
    return build_services(settings, llm=llm, web=web or FakeWebTools(), today=TODAY)
