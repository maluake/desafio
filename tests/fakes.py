"""Test doubles: a scripted Anthropic Messages API and offline web tools."""

from __future__ import annotations

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_use(name: str, args: dict[str, Any], id_: str = "tu_1") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=args)


def response(*blocks: Any, stop_reason: str | None = None) -> SimpleNamespace:
    if stop_reason is None:
        stop_reason = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


def json_response(data: dict[str, Any]) -> SimpleNamespace:
    return response(text_block(json.dumps(data)))


class FakeMessages:
    """Mimics ``client.beta.messages``. ``handler(kwargs, call_index)`` returns a response.

    Every call's kwargs are recorded in ``calls`` for assertions.
    """

    def __init__(self, handler: Callable[[dict[str, Any], int], Any]) -> None:
        self.handler = handler
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        out = self.handler(kwargs, len(self.calls) - 1)
        if isinstance(out, Exception):
            raise out
        return out


class ScriptedMessages(FakeMessages):
    """Returns the given responses in order."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        super().__init__(lambda kw, i: self._responses[i])


class FakeWebTools:
    """Deterministic replacements for weather / FX / web search."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def web_search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        self.calls.append(("web_search", {"query": query}))
        if self.fail:
            raise RuntimeError("network down")
        return {
            "query": query,
            "provider": "fake",
            "results": [{"title": "Example result", "url": "https://example.com", "snippet": "Example snippet about " + query}],
        }

    def get_weather(self, city: str, day: str | None = None) -> dict[str, Any]:
        self.calls.append(("get_weather", {"city": city, "day": day}))
        if self.fail:
            raise RuntimeError("network down")
        return {
            "city": city, "region": "RS", "country": "Brasil", "date": "2026-10-16", "condition": "rain showers",
            "temp_min_c": 14.2, "temp_max_c": 22.8, "precipitation_probability_pct": 80, "precipitation_mm": 12.5,
            "wind_max_kmh": 25.0, "source": "Open-Meteo (open-meteo.com)",
        }

    def get_exchange_rate(self, base: str, quote: str = "BRL") -> dict[str, Any]:
        self.calls.append(("get_exchange_rate", {"base": base, "quote": quote}))
        if self.fail:
            raise RuntimeError("network down")
        return {"base": base.upper(), "quote": quote, "rate": 6.1234, "date": "2026-10-15", "source": "Frankfurter / ECB"}
