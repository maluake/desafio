"""Public API contracts and internal data-transfer objects shared by the agents."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class AgentName(StrEnum):
    ROUTER = "router"
    KNOWLEDGE = "knowledge"
    SUPPORT = "support"
    ESCALATION = "escalation"
    GUARDRAIL = "guardrail"


# ----------------------------------------------------------------------------- API


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's message")
    user_id: str = Field(..., min_length=1, max_length=128, description="Authenticated user identifier")

    @field_validator("message", "user_id")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


class Source(BaseModel):
    title: str
    url: str | None = None
    snippet: str | None = None
    score: float | None = None


class ToolCallRecord(BaseModel):
    agent: AgentName
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    latency_ms: float = 0.0
    error: str | None = None


class RouteDecision(BaseModel):
    """Output of the Router Agent."""

    plan: list[AgentName] = Field(description="Ordered agents to run")
    intent: str = Field(description="Short machine-friendly intent label")
    confidence: float = Field(ge=0.0, le=1.0)
    language: str = Field(default="en", description="ISO code of the user's language")
    reasoning: str = ""
    needs_human: bool = False


class GuardrailReport(BaseModel):
    input_flags: list[str] = Field(default_factory=list)
    output_flags: list[str] = Field(default_factory=list)
    pii_redacted: bool = False
    blocked: bool = False


class ChatResponse(BaseModel):
    trace_id: str
    user_id: str
    response: str
    agent: AgentName = Field(description="Agent that produced the final answer")
    route: RouteDecision | None = None
    agents_involved: list[AgentName] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    escalated: bool = False
    ticket_id: str | None = None
    guardrails: GuardrailReport = Field(default_factory=GuardrailReport)
    mode: str = Field(description="'llm' when Claude answered, 'offline' for deterministic fallback")
    latency_ms: float = 0.0
    trace: list[dict[str, Any]] | None = None


class FeedbackRequest(BaseModel):
    trace_id: str
    user_id: str
    rating: int = Field(ge=-1, le=1, description="-1 thumbs down, 0 neutral, 1 thumbs up")
    comment: str | None = None


# ------------------------------------------------------------------------ Internal


class AgentResult(BaseModel):
    agent: AgentName
    answer: str
    sources: list[Source] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    confidence: float = 1.0
    needs_human: bool = False
    handoff_reason: str | None = None
    ticket_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
