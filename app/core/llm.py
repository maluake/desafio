"""Thin, observable wrapper around the Anthropic Messages API.

Responsibilities:
* one place to set model / effort / refusal fallback / timeouts / retries;
* translate SDK exceptions into a small internal hierarchy so agents can degrade
  gracefully (the orchestrator falls back to deterministic policies on ``LLMError``);
* emit token / call metrics and a span per call.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

import anthropic

from app.config import Settings
from app.core.observability import LLM_CALLS, LLM_TOKENS, get_logger, span

log = get_logger(__name__)

REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """Any failure talking to the model (network, auth, rate limit, 5xx, refusal...)."""


class LLMRefusal(LLMError):
    """The model declined to answer (``stop_reason == 'refusal'``)."""


class MessagesAPI(Protocol):  # pragma: no cover - structural type for test doubles
    def create(self, **kwargs: Any) -> Any: ...


class LLMClient:
    def __init__(self, settings: Settings, messages_api: MessagesAPI | None = None) -> None:
        self.settings = settings
        if messages_api is not None:
            self._messages = messages_api
        else:
            client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key,
                timeout=settings.llm_timeout_s,
                max_retries=settings.llm_max_retries,
            )
            self._messages = client.beta.messages

    def create(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        effort: str | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        tool_choice: dict[str, Any] | None = None,
        name: str = "llm",
    ) -> Any:
        model = model or self.settings.llm_model
        output_config: dict[str, Any] = {"effort": effort or self.settings.agent_effort}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
            "system": system,
            "messages": messages,
            "output_config": output_config,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if self.settings.llm_refusal_fallback:
            kwargs["betas"] = [REFUSAL_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"

        with span(f"llm.{name}", model=model, effort=output_config["effort"]) as s:
            try:
                response = self._messages.create(**kwargs)
            except anthropic.AuthenticationError as exc:
                LLM_CALLS.labels(model, "auth_error").inc()
                raise LLMError(f"authentication failed: {exc.message}") from exc
            except anthropic.RateLimitError as exc:
                LLM_CALLS.labels(model, "rate_limited").inc()
                raise LLMError("rate limited") from exc
            except anthropic.BadRequestError as exc:
                LLM_CALLS.labels(model, "bad_request").inc()
                raise LLMError(f"bad request: {exc.message}") from exc
            except anthropic.APIStatusError as exc:
                LLM_CALLS.labels(model, f"http_{exc.status_code}").inc()
                raise LLMError(f"api error {exc.status_code}") from exc
            except anthropic.APIConnectionError as exc:
                LLM_CALLS.labels(model, "connection_error").inc()
                raise LLMError("connection error") from exc

            usage = getattr(response, "usage", None)
            if usage is not None:
                in_tok = getattr(usage, "input_tokens", 0) or 0
                out_tok = getattr(usage, "output_tokens", 0) or 0
                LLM_TOKENS.labels(model, "input").inc(in_tok)
                LLM_TOKENS.labels(model, "output").inc(out_tok)
                s.attributes.update(input_tokens=in_tok, output_tokens=out_tok)
            s.attributes["stop_reason"] = response.stop_reason

            if response.stop_reason == "refusal":
                LLM_CALLS.labels(model, "refusal").inc()
                details = getattr(response, "stop_details", None)
                raise LLMRefusal(f"model refused: {getattr(details, 'category', None)}")
            LLM_CALLS.labels(model, "ok").inc()
            return response

    def create_json(self, *, json_schema: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Structured output call: the first text block is guaranteed to be valid JSON."""
        response = self.create(json_schema=json_schema, **kwargs)
        text = text_of(response)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:  # defensive: should not happen with output_config.format
            raise LLMError(f"invalid JSON from model: {text[:200]}") from exc


def text_of(response: Any) -> str:
    return "".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()
