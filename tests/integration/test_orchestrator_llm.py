"""End-to-end orchestration in LLM mode with a scripted model.

The fake model dispatches on the system prompt (router / knowledge / support / composer)
so we exercise the real prompts, tool schemas, tool loop, composition and escalation.
"""

import json

from app.agents.orchestrator import Orchestrator
from tests.conftest import make_llm_services
from tests.fakes import FakeMessages, json_response, response, text_block, tool_use


def scripted_model(plan, intent="account_data", confidence=0.95):
    def handler(kw, i):
        system = kw["system"]
        last = kw["messages"][-1]["content"]
        has_tool_result = isinstance(last, list) and last and last[0].get("type") == "tool_result"
        if system.startswith("You are the Router"):
            return json_response({"intent": intent, "plan": plan, "confidence": confidence, "language": "en",
                                  "needs_human": "escalation" in plan, "reasoning": "test"})
        if "Customer Support Agent" in system:
            if not has_tool_result:
                return response(tool_use("get_settlement_forecast", {"sale_date": "yesterday"}, "s1"))
            data = json.loads(last[0]["content"])
            return response(text_block(f"Your debit sales arrive on {data['deposits'][0]['expected_date']}."))
        if "Knowledge Agent" in system:
            if not has_tool_result:
                return response(tool_use("search_knowledge_base", {"query": "antecipação"}, "k1"))
            return response(text_block("Advance lets you receive earlier [1]."))
        if "merge the answers" in system:
            return response(text_block("MERGED: debit arrives 2026-10-15; advance lets you receive earlier [1]."))
        if "hand-off notes" in system:
            return json_response({"summary": "Customer wants a human.", "priority": "normal", "category": "general"})
        raise AssertionError(f"unexpected call: {system[:60]}")

    return handler


def test_llm_support_flow(settings):
    api = FakeMessages(scripted_model(["support"]))
    o = Orchestrator(make_llm_services(settings, api))
    r = o.handle("When will the money from yesterday's sales be deposited?", "cliente1988")
    assert r.mode == "llm" and r.agent.value == "support"
    assert "2026-10-15" in r.response
    assert [c.tool for c in r.tool_calls] == ["get_settlement_forecast"]
    support_call = next(c for c in api.calls if "Customer Support Agent" in c["system"])
    assert "2026-10-14" in support_call["system"]  # 'yesterday' resolved in the prompt
    assert {t["name"] for t in support_call["tools"]} >= {"get_settlement_forecast", "get_terminal_status", "open_support_ticket"}


def test_llm_multi_agent_composition(settings):
    api = FakeMessages(scripted_model(["support", "knowledge"]))
    o = Orchestrator(make_llm_services(settings, api))
    r = o.handle("When will yesterday's sales be deposited and how does antecipação work?", "cliente1988")
    assert r.response.startswith("MERGED")
    assert [a.value for a in r.agents_involved] == ["router", "support", "knowledge"]
    knowledge_call = next(c for c in api.calls if "Knowledge Agent" in c["system"])
    assert "<context_from_other_agents>" in knowledge_call["messages"][-1]["content"] or any(
        "<context_from_other_agents>" in str(m["content"]) for m in knowledge_call["messages"]
    )


def test_llm_escalation(settings):
    api = FakeMessages(scripted_model(["escalation"], intent="human_handoff"))
    o = Orchestrator(make_llm_services(settings, api))
    r = o.handle("Can someone from Getnet call me?", "cliente1988")
    assert r.escalated and r.ticket_id and r.ticket_id in r.response


def test_low_confidence_triggers_handoff(settings):
    def handler(kw, i):
        if kw["system"].startswith("You are the Router"):
            return json_response({"intent": "product_info", "plan": ["knowledge"], "confidence": 0.2, "language": "en",
                                  "needs_human": False, "reasoning": "vague"})
        if "Knowledge Agent" in kw["system"]:
            return response(text_block("I am not sure what you mean."))  # no retrieval -> ungrounded, low confidence
        return json_response({"summary": "unclear request", "priority": "normal", "category": "general"})

    o = Orchestrator(make_llm_services(settings, FakeMessages(handler)))
    r = o.handle("the thing with the stuff", "cliente1988")
    assert r.escalated and "ungrounded" in r.guardrails.output_flags


def test_memory_is_passed_to_next_turn(settings):
    api = FakeMessages(scripted_model(["support"]))
    o = Orchestrator(make_llm_services(settings, api))
    o.handle("When will the money from yesterday's sales be deposited?", "cliente1988")
    o.handle("And the credit ones?", "cliente1988")
    router_calls = [c for c in api.calls if c["system"].startswith("You are the Router")]
    assert "<recent_conversation>" in router_calls[1]["messages"][0]["content"]
