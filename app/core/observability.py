"""Structured logging, lightweight tracing and Prometheus metrics.

Design notes
------------
* Every request gets a ``trace_id`` stored in a ``ContextVar`` so that any log line
  emitted anywhere in the call stack (agents, tools, RAG) is correlated automatically.
* ``span()`` records nested, timed operations (agent runs, tool calls, LLM calls) into the
  current ``Trace``. The trace is returned in the API response (debug) and logged at the end
  of the request, which makes it trivial to ship to Langfuse / OpenTelemetry later: the span
  model (name, parent, attributes, duration, status) maps 1:1 to an OTel span.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from prometheus_client import Counter, Histogram

# ------------------------------------------------------------------------- metrics

REQUESTS = Counter("chat_requests_total", "Chat requests", ["final_agent", "mode", "escalated"])
REQUEST_LATENCY = Histogram(
    "chat_request_latency_seconds",
    "End-to-end chat latency",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64),
)
ROUTES = Counter("router_decisions_total", "Router decisions", ["intent", "first_agent"])
ROUTER_CONFIDENCE = Histogram(
    "router_confidence", "Router confidence", buckets=(0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)
AGENT_LATENCY = Histogram("agent_latency_seconds", "Agent latency", ["agent"])
AGENT_ERRORS = Counter("agent_errors_total", "Agent failures (fell back or errored)", ["agent"])
TOOL_CALLS = Counter("tool_calls_total", "Tool invocations", ["agent", "tool", "status"])
TOOL_LATENCY = Histogram("tool_latency_seconds", "Tool latency", ["tool"])
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens", ["model", "kind"])
LLM_CALLS = Counter("llm_calls_total", "LLM calls", ["model", "status"])
GUARDRAIL_EVENTS = Counter("guardrail_events_total", "Guardrail triggers", ["stage", "flag"])
ESCALATIONS = Counter("escalations_total", "Human handoffs", ["reason"])
RAG_RETRIEVALS = Counter("rag_retrievals_total", "RAG retrievals", ["hit"])
FEEDBACK = Counter("user_feedback_total", "User feedback", ["rating"])

# ------------------------------------------------------------------------- tracing

_trace_var: contextvars.ContextVar[Trace | None] = contextvars.ContextVar("trace", default=None)
_span_stack: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar("span_stack", default=())


@dataclass
class Span:
    span_id: str
    name: str
    parent_id: str | None
    start: float
    attributes: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "duration_ms": round(self.duration_ms or 0.0, 2),
            "status": self.status,
            "attributes": self.attributes,
        }


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    spans: list[Span] = field(default_factory=list)

    def to_list(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.spans]


def start_trace() -> Trace:
    trace = Trace()
    _trace_var.set(trace)
    _span_stack.set(())
    return trace


def current_trace() -> Trace | None:
    return _trace_var.get()


def current_trace_id() -> str | None:
    t = _trace_var.get()
    return t.trace_id if t else None


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    trace = _trace_var.get()
    stack = _span_stack.get()
    s = Span(
        span_id=uuid.uuid4().hex[:16],
        name=name,
        parent_id=stack[-1] if stack else None,
        start=time.perf_counter(),
        attributes=dict(attributes),
    )
    token = _span_stack.set((*stack, s.span_id))
    try:
        yield s
    except Exception as exc:
        s.status = "error"
        s.attributes["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        s.duration_ms = (time.perf_counter() - s.start) * 1000
        _span_stack.reset(token)
        if trace is not None:
            trace.spans.append(s)


# ------------------------------------------------------------------------- logging


class JsonFormatter(logging.Formatter):
    _RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": current_trace_id(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpx2", "httpcore", "anthropic", "primp", "ddgs"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
