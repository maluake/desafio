"""LLM client + agent tool-use loop, driven by a scripted Messages API (no network)."""

import anthropic
import httpx2
import pytest

from app.agents.knowledge import KnowledgeAgent
from app.agents.router import RouterAgent
from app.core.agent_base import AgentContext
from app.core.llm import REFUSAL_FALLBACK_BETA, LLMClient, LLMError
from app.schemas import AgentName
from tests.fakes import FakeMessages, ScriptedMessages, json_response, response, text_block, tool_use


def test_request_shape(settings):
    api = ScriptedMessages([response(text_block("hi"))])
    LLMClient(settings, api).create(system="s", messages=[{"role": "user", "content": "x"}], effort="low")
    kw = api.calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"] == {"effort": "low"}
    assert kw["betas"] == [REFUSAL_FALLBACK_BETA] and kw["fallbacks"] == "default"
    assert "thinking" not in kw and "temperature" not in kw


def test_structured_output(settings):
    api = ScriptedMessages([json_response({"a": 1})])
    out = LLMClient(settings, api).create_json(system="s", messages=[], json_schema={"type": "object"})
    assert out == {"a": 1}
    assert api.calls[0]["output_config"]["format"]["type"] == "json_schema"


def _status_error(cls, code):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx2.Response(code, request=req), body=None)


@pytest.mark.parametrize(
    "exc",
    [
        lambda: _status_error(anthropic.RateLimitError, 429),
        lambda: _status_error(anthropic.InternalServerError, 500),
        lambda: _status_error(anthropic.AuthenticationError, 401),
        lambda: anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x")),
    ],
)
def test_errors_are_wrapped(settings, exc):
    api = FakeMessages(lambda kw, i: exc())
    with pytest.raises(LLMError):
        LLMClient(settings, api).create(system="s", messages=[])


def test_refusal_raises(settings):
    api = ScriptedMessages([response(text_block(""), stop_reason="refusal")])
    with pytest.raises(LLMError):
        LLMClient(settings, api).create(system="s", messages=[])


def test_router_llm(settings):
    api = ScriptedMessages([json_response({
        "intent": "account_data", "plan": ["support", "support", "knowledge"], "confidence": 1.4,
        "language": "pt", "needs_human": False, "reasoning": "x",
    })])
    decision, mode = RouterAgent(settings, LLMClient(settings, api)).route("quando cai meu dinheiro?")
    assert mode == "llm"
    assert decision.plan == [AgentName.SUPPORT, AgentName.KNOWLEDGE]  # deduped
    assert decision.confidence == 1.0  # clamped
    assert api.calls[0]["output_config"]["effort"] == "low"


def test_router_falls_back_to_rules(settings):
    api = FakeMessages(lambda kw, i: _status_error(anthropic.InternalServerError, 500))
    decision, mode = RouterAgent(settings, LLMClient(settings, api)).route("What's the euro exchange rate today?")
    assert mode == "offline" and decision.plan == [AgentName.KNOWLEDGE]


def test_knowledge_tool_loop(services):
    api = ScriptedMessages([
        response(text_block("Let me check."), tool_use("search_knowledge_base", {"query": "crediário parcelas"}, "tu_1")),
        response(text_block("Up to 48 installments [1].")),
    ])
    agent = KnowledgeAgent(services.settings, LLMClient(services.settings, api), services)
    res = agent.run(AgentContext(message="How many installments with crediário?", user_id="cliente1988"))
    assert res.answer == "Up to 48 installments [1]." and res.data["mode"] == "llm"
    assert [c.tool for c in res.tool_calls] == ["search_knowledge_base"]
    assert res.sources and "Crediário" in res.sources[0].title
    # 2nd call carries assistant tool_use + ONE user message with the tool_result
    second = api.calls[1]["messages"]
    assert second[-2]["role"] == "assistant"
    assert second[-1]["content"][0]["type"] == "tool_result" and second[-1]["content"][0]["tool_use_id"] == "tu_1"


def test_parallel_tool_results_in_one_message(services):
    api = ScriptedMessages([
        response(tool_use("get_weather", {"city": "Porto Alegre"}, "a"), tool_use("get_exchange_rate", {"base": "EUR"}, "b")),
        response(text_block("done")),
    ])
    agent = KnowledgeAgent(services.settings, LLMClient(services.settings, api), services)
    agent.run(AgentContext(message="weather and euro", user_id="u"))
    results = api.calls[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]


def test_tool_errors_are_reported_to_model(services):
    api = ScriptedMessages([response(tool_use("get_weather", {}, "a")), response(text_block("sorry"))])
    agent = KnowledgeAgent(services.settings, LLMClient(services.settings, api), services)
    agent.run(AgentContext(message="weather", user_id="u"))
    assert api.calls[1]["messages"][-1]["content"][0]["is_error"] is True


def test_iteration_budget_forces_final_answer(services):
    loop = response(tool_use("search_knowledge_base", {"query": "x"}))
    final = response(text_block("final"))
    api = FakeMessages(lambda kw, i: final if kw.get("tool_choice") == {"type": "none"} else loop)
    services.settings.agent_max_tool_iterations = 2
    agent = KnowledgeAgent(services.settings, LLMClient(services.settings, api), services)
    res = agent.run(AgentContext(message="x", user_id="u"))
    assert res.answer == "final" and len(api.calls) == 3


def test_llm_failure_degrades_to_offline(services):
    api = FakeMessages(lambda kw, i: _status_error(anthropic.InternalServerError, 503))
    agent = KnowledgeAgent(services.settings, LLMClient(services.settings, api), services)
    res = agent.run(AgentContext(message="How many installments can I split a sale into with the crediário?", user_id="u"))
    assert res.data["mode"] == "offline" and "48" in res.answer


def test_handoff_tool_sets_needs_human(services):
    api = ScriptedMessages([
        response(tool_use("request_human_handoff", {"reason": "cannot resolve"})),
        response(text_block("A specialist will contact you.")),
    ])
    agent = KnowledgeAgent(services.settings, LLMClient(services.settings, api), services)
    res = agent.run(AgentContext(message="x", user_id="u"))
    assert res.needs_human and res.handoff_reason == "cannot resolve"
