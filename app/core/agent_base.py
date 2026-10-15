"""Base class for tool-using agents.

Each specialised agent declares a system prompt, a toolset and an offline policy.
``run()`` executes the Claude tool-use loop and, if the LLM is unavailable or fails,
degrades to ``run_offline()`` - a deterministic policy that calls the *same* tools.
This keeps the service useful during an LLM outage and makes integration tests
fully deterministic.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.core.llm import LLMClient, LLMError, text_of
from app.core.observability import AGENT_ERRORS, AGENT_LATENCY, get_logger, span
from app.core.tools import Tool, ToolContext, ToolExecutor, to_tool_content
from app.schemas import AgentName, AgentResult, RouteDecision, Source

log = get_logger(__name__)

HANDOFF_TOOL_NAME = "request_human_handoff"


@dataclass
class AgentContext:
    message: str  # already PII-redacted by the input guardrail
    user_id: str
    language: str = "en"
    history: list[dict[str, str]] = field(default_factory=list)
    prior_results: list[AgentResult] = field(default_factory=list)
    route: RouteDecision | None = None
    today: str | None = None  # injected clock (ISO date) for reproducibility


def handoff_tool() -> Tool:
    """Every specialist can ask for a human when it cannot resolve the request confidently."""

    def _handler(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        ctx.extras["handoff_reason"] = args["reason"]
        return {"status": "handoff_requested", "note": "A human specialist will take over after your reply."}

    return Tool(
        name=HANDOFF_TOOL_NAME,
        description=(
            "Request a human specialist. Use ONLY when you cannot resolve the request with the "
            "available tools and information, when the customer explicitly asks for a person, or "
            "when the issue involves suspected fraud, legal threats or account security. "
            "After calling it, tell the customer that a specialist will contact them."
        ),
        input_schema={
            "type": "object",
            "properties": {"reason": {"type": "string", "description": "Why a human is needed"}},
            "required": ["reason"],
        },
        handler=_handler,
    )


class BaseAgent(ABC):
    name: AgentName

    def __init__(self, settings: Settings, llm: LLMClient | None, services: Any) -> None:
        self.settings = settings
        self.llm = llm
        self.services = services

    # ----------------------------------------------------------------- to implement
    @abstractmethod
    def system_prompt(self, ctx: AgentContext) -> str: ...

    @abstractmethod
    def tools(self) -> list[Tool]: ...

    @abstractmethod
    def run_offline(self, ctx: AgentContext, executor: ToolExecutor) -> AgentResult: ...

    # ----------------------------------------------------------------- shared logic
    def run(self, ctx: AgentContext) -> AgentResult:
        start = time.perf_counter()
        tool_ctx = ToolContext(
            user_id=ctx.user_id, services=self.services, extras={"intent": ctx.route.intent if ctx.route else None}
        )
        with span(f"agent.{self.name.value}") as s:
            executor = ToolExecutor(self.name, self.tools(), tool_ctx)
            result: AgentResult | None = None
            if self.llm is not None:
                try:
                    result = self._run_llm(ctx, executor)
                    s.attributes["mode"] = "llm"
                except LLMError as exc:
                    AGENT_ERRORS.labels(self.name.value).inc()
                    log.warning("LLM failed, degrading to offline policy", extra={"agent": self.name, "error": str(exc)})
                    s.attributes["llm_error"] = str(exc)
            if result is None:
                result = self.run_offline(ctx, executor)
                result.data["mode"] = "offline"
                s.attributes["mode"] = "offline"
            else:
                result.data["mode"] = "llm"

            reason = tool_ctx.extras.get("handoff_reason")
            if reason:
                result.needs_human = True
                result.handoff_reason = result.handoff_reason or reason
            result.tool_calls = executor.records
            s.attributes.update(needs_human=result.needs_human, tools=[r.tool for r in executor.records])
        AGENT_LATENCY.labels(self.name.value).observe(time.perf_counter() - start)
        return result

    def build_user_turn(self, ctx: AgentContext) -> str:
        parts = []
        if ctx.prior_results:
            parts.append("<context_from_other_agents>")
            for r in ctx.prior_results:
                parts.append(f'<agent name="{r.agent.value}">\n{r.answer}\n</agent>')
            parts.append("</context_from_other_agents>")
        parts.append(f"<customer_message>\n{ctx.message}\n</customer_message>")
        return "\n".join(parts)

    def _run_llm(self, ctx: AgentContext, executor: ToolExecutor) -> AgentResult:
        assert self.llm is not None
        tool_defs = [t.to_anthropic() for t in self.tools()]
        messages: list[dict[str, Any]] = [*ctx.history, {"role": "user", "content": self.build_user_turn(ctx)}]
        system = self.system_prompt(ctx)
        collected: list[Any] = []

        for iteration in range(self.settings.agent_max_tool_iterations):
            response = self.llm.create(
                system=system, messages=messages, tools=tool_defs, name=f"{self.name.value}.turn{iteration}"
            )
            tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
            if response.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": response.content})
                continue
            if not tool_uses:
                return self.finalize(text_of(response), executor, collected)

            # Keep the full assistant content (incl. thinking blocks) to preserve reasoning continuity.
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in tool_uses:
                output, ok = executor.run(block.name, block.input)
                collected.append((block.name, output, ok))
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": to_tool_content(output),
                        **({} if ok else {"is_error": True}),
                    }
                )
            # All results go back in ONE user message (keeps parallel tool use working).
            messages.append({"role": "user", "content": results})

        # Iteration budget exhausted: ask for a final answer without tools.
        response = self.llm.create(
            system=system + "\nYou have used all tool calls available. Answer now with what you have.",
            messages=messages,
            tools=tool_defs,
            tool_choice={"type": "none"},
            name=f"{self.name.value}.final",
        )
        return self.finalize(text_of(response), executor, collected)

    def finalize(self, answer: str, executor: ToolExecutor, collected: list[Any]) -> AgentResult:
        """Hook for agents to attach sources / metadata to the LLM answer."""
        return AgentResult(agent=self.name, answer=answer)


def dedupe_sources(sources: list[Source]) -> list[Source]:
    seen: set[str] = set()
    out: list[Source] = []
    for s in sources:
        key = s.url or s.title
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out
