from app.core.tools import Tool, ToolContext, ToolExecutor
from app.schemas import AgentName


def _tool(handler=lambda ctx, a: {"ok": True, "user": ctx.user_id}):
    return Tool(
        "t", "test",
        {"type": "object", "properties": {"n": {"type": "integer"}, "mode": {"type": "string", "enum": ["a", "b"]}}, "required": ["n"]},
        handler,
    )


def test_validation_and_unknown_keys_dropped():
    ex = ToolExecutor(AgentName.SUPPORT, [_tool()], ToolContext("u1", None))
    out, ok = ex.run("t", {"n": 1, "user_id": "attacker"})
    assert ok and out["user"] == "u1"  # model cannot override the authenticated user


def test_errors_are_returned_not_raised():
    ex = ToolExecutor(AgentName.SUPPORT, [_tool(lambda c, a: 1 / 0)], ToolContext("u1", None))
    assert ex.run("t", {"n": 1}) == ({"error": "ZeroDivisionError: division by zero"}, False)
    assert ex.run("t", {})[1] is False  # missing required
    assert ex.run("t", {"n": "x"})[1] is False  # wrong type
    assert ex.run("t", {"n": 1, "mode": "z"})[1] is False  # enum
    assert ex.run("nope", {})[1] is False  # unknown tool
    assert [r.ok for r in ex.records] == [False] * 5


def test_support_tools_are_scoped_to_authenticated_user(services):
    from app.agents.support import SupportAgent

    agent = SupportAgent(services.settings, None, services)
    schemas = {t.name: t.input_schema for t in agent.tools()}
    assert all("user_id" not in s.get("properties", {}) for s in schemas.values())
    ex = ToolExecutor(AgentName.SUPPORT, agent.tools(), ToolContext("cliente2024", services))
    out, ok = ex.run("get_customer_profile", {"user_id": "cliente1988"})
    assert ok and out["merchant_name"] == "Oficina Auto Center Sul"
