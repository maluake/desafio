"""HTTP API (FastAPI).

Endpoints
---------
POST /chat              main entry point: {"message", "user_id"} -> ChatResponse
POST /feedback          thumbs up/down per trace (feeds quality dashboards / eval datasets)
GET  /health            liveness + dependency summary
GET  /metrics           Prometheus metrics
POST /admin/reindex     rebuild the knowledge base (seed or crawl)
GET  /tickets/{user_id} hand-off/support tickets created for a user (demo/ops)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import __version__
from app.agents.orchestrator import Orchestrator
from app.agents.prompts import PROMPT_VERSION
from app.config import get_settings
from app.container import Services, build_services
from app.core.observability import FEEDBACK, configure_logging, get_logger
from app.rag.ingest import build_index
from app.schemas import ChatRequest, ChatResponse, FeedbackRequest

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    if not hasattr(app.state, "services"):
        app.state.services = build_services(settings)
        app.state.orchestrator = Orchestrator(app.state.services)
    log.info("service started", extra={"version": __version__, "prompt_version": PROMPT_VERSION})
    yield


def create_app(services: Services | None = None) -> FastAPI:
    app = FastAPI(
        title="Getnet Multi-Agent Support",
        version=__version__,
        description="Router + Knowledge (RAG) + Customer Support + Human Escalation agents.",
        lifespan=lifespan,
    )
    if services is not None:
        app.state.services = services
        app.state.orchestrator = Orchestrator(services)

    @app.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
    async def chat(req: ChatRequest, request: Request) -> ChatResponse:
        orchestrator: Orchestrator = request.app.state.orchestrator
        return await run_in_threadpool(orchestrator.handle, req.message, req.user_id)

    @app.post("/feedback")
    async def feedback(req: FeedbackRequest) -> dict[str, str]:
        FEEDBACK.labels(str(req.rating)).inc()
        log.info("feedback", extra=req.model_dump())
        return {"status": "recorded"}

    @app.get("/health")
    async def health(request: Request) -> dict:
        s: Services = request.app.state.services
        return {
            "status": "ok",
            "version": __version__,
            "prompt_version": PROMPT_VERSION,
            "llm": {"provider": "anthropic" if s.llm else "offline", "model": s.settings.llm_model if s.llm else None},
            "knowledge_base": {"chunks": len(s.kb.chunks), "documents": len({c.doc_id for c in s.kb.chunks})},
        }

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/admin/reindex")
    async def reindex(request: Request, mode: Literal["seed", "crawl"] = "seed") -> dict:
        s: Services = request.app.state.services
        store = await run_in_threadpool(build_index, s.settings, mode)
        s.kb = store
        return {"status": "ok", "chunks": len(store.chunks), "mode": mode}

    @app.get("/tickets/{user_id}")
    async def tickets(user_id: str, request: Request) -> list[dict]:
        s: Services = request.app.state.services
        return s.tickets.list_for(user_id)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return JSONResponse(status_code=500, content={"detail": "internal error", "type": type(exc).__name__})

    return app


app = create_app()
