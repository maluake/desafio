"""Tool abstraction shared by every agent.

A ``Tool`` couples the JSON schema the LLM sees with the Python handler that runs it.
Handlers receive a ``ToolContext`` carrying the *authenticated* ``user_id``: tools never
accept a user id from the model, so a prompt-injected "show me user X's sales" cannot
reach another customer's data (confused-deputy protection).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.observability import TOOL_CALLS, TOOL_LATENCY, get_logger, span
from app.schemas import AgentName, ToolCallRecord

log = get_logger(__name__)

_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


class ToolInputError(ValueError):
    pass


@dataclass
class ToolContext:
    user_id: str
    services: Any  # app.container.Services - typed loosely to avoid import cycles
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[ToolContext, dict[str, Any]], Any]

    def to_anthropic(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def validate(self, args: Any) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ToolInputError("tool input must be an object")
        props: dict[str, Any] = self.input_schema.get("properties", {})
        for key in self.input_schema.get("required", []):
            if key not in args or args[key] in (None, ""):
                raise ToolInputError(f"missing required argument '{key}'")
        clean: dict[str, Any] = {}
        for key, value in args.items():
            if key not in props:
                continue  # drop unknown keys instead of failing hard
            expected = props[key].get("type")
            if expected and value is not None and not isinstance(value, _JSON_TYPES.get(expected, object)):
                raise ToolInputError(f"argument '{key}' must be of type {expected}")
            if "enum" in props[key] and value is not None and value not in props[key]["enum"]:
                raise ToolInputError(f"argument '{key}' must be one of {props[key]['enum']}")
            clean[key] = value
        return clean


class ToolExecutor:
    """Runs tools with validation, timing, metrics, tracing and error capture."""

    def __init__(self, agent: AgentName, tools: list[Tool], ctx: ToolContext) -> None:
        self.agent = agent
        self.tools = {t.name: t for t in tools}
        self.ctx = ctx
        self.records: list[ToolCallRecord] = []

    def run(self, name: str, args: Any) -> tuple[Any, bool]:
        """Returns ``(result, ok)``. Never raises: errors are returned to the model as data."""
        start = time.perf_counter()
        tool = self.tools.get(name)
        ok, error, result = True, None, None
        safe_args = args if isinstance(args, dict) else {}
        with span(f"tool.{name}", agent=self.agent.value, arguments=safe_args) as s:
            try:
                if tool is None:
                    raise ToolInputError(f"unknown tool '{name}'")
                clean = tool.validate(args)
                result = tool.handler(self.ctx, clean)
            except Exception as exc:  # tool failures must not crash the agent loop
                ok, error = False, f"{type(exc).__name__}: {exc}"
                result = {"error": error}
                s.status = "error"
                log.warning("tool failed", extra={"tool": name, "error": error})
        latency = (time.perf_counter() - start) * 1000
        TOOL_CALLS.labels(self.agent.value, name, "ok" if ok else "error").inc()
        TOOL_LATENCY.labels(name).observe(latency / 1000)
        self.records.append(
            ToolCallRecord(
                agent=self.agent, tool=name, arguments=safe_args, ok=ok, latency_ms=round(latency, 2), error=error
            )
        )
        return result, ok


def to_tool_content(result: Any) -> str:
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
