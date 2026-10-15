"""Router Agent: entry point that classifies the request and plans the agent sequence.

LLM path: a single structured-output call (JSON schema enforced by the API) at low effort.
Offline path: transparent weighted keyword rules, also used when the LLM call fails.
"""

from __future__ import annotations

import re
from typing import Any

from app.agents.prompts import ROUTER_SCHEMA, ROUTER_SYSTEM
from app.config import Settings
from app.core.language import detect_language
from app.core.llm import LLMClient, LLMError
from app.core.observability import AGENT_ERRORS, ROUTER_CONFIDENCE, ROUTES, get_logger, span
from app.rag.text import normalize
from app.schemas import AgentName, RouteDecision

log = get_logger(__name__)

# ---------------------------------------------------------------- offline rules
_PERSONAL_TIME = r"(yesterday|today|last (week|month)|this (week|month)|ontem|hoje|semana passada|mes passado|esta semana)"
_MONEY = r"(money|deposit|deposited|paid|payout|receive|dinheiro|deposito|depositad|cai|caiu|recebo|receber|pagamento|valor)"
_MY_DEVICE = r"(my|minha|meu|the)\s+(card\s+)?(machine|maquininha|maquina|terminal|pos|get smart|get classica|get mini)"
_PROBLEM = r"(won.?t|doesn.?t|does not|can.?t|cannot|not|nao|error|erro|declin|negad|recusad|fail|falha|problem|problema|offline|connect|conecta|showing|mostrando)"
_MY_DATA = r"\b(my|minhas?|meus?)\s+(sales|vendas|deposits?|depositos|receivables|recebiveis|plan|plano|account balance|saldo|transactions|transacoes|tickets?|chamados?)\b"

_KNOWLEDGE_HINTS = [
    r"\b(what|how|which|difference|does|do i|can i|is there|explain|o que|como|qual|quais|diferenca|posso|preciso|funciona)\b",
    r"\b(get smart|get classica|get mini|pix|link de pagamento|payment link|crediario|antecipacao|receivables advance|whatsapp|taxa|fee|rent|aluguel|e-?commerce)\b",
]
_GENERAL = r"\b(weather|forecast|temperature|rain|clima|previsao|tempo|chuva|temperatura|exchange rate|cambio|cotacao|dollar|dolar|euro|usd|eur|news|noticias)\b"
_HUMAN = r"\b(human|humano|atendente|real person|pessoa real|operator|operador|talk to someone|falar com alguem)\b"
_GREETING = r"^(hi|hello|hey|oi|ola|bom dia|boa tarde|boa noite|thanks|thank you|obrigad[oa]|valeu)[\s!.,]*$"
_UNSUPPORTED = r"\b(write (me )?(a|an) (poem|essay|story)|homework|dever de casa|redacao|poema|diagnos(e|tico) medic|recipe|receita de bolo)\b"


def offline_route(message: str, language: str) -> RouteDecision:
    m = normalize(message)
    if re.search(_GREETING, m):
        return RouteDecision(plan=[AgentName.KNOWLEDGE], intent="greeting", confidence=0.95, language=language, reasoning="greeting")
    if re.search(_HUMAN, m):
        return RouteDecision(
            plan=[AgentName.ESCALATION], intent="human_handoff", confidence=0.9, language=language,
            needs_human=True, reasoning="customer asked for a human",
        )
    if re.search(_UNSUPPORTED, m):
        return RouteDecision(plan=[], intent="unsupported", confidence=0.8, language=language, reasoning="out of scope")

    support = 0.0
    if re.search(_PERSONAL_TIME, m) and re.search(_MONEY + "|sales|vendas", m):
        support += 2.0
    if re.search(_MY_DEVICE, m) and re.search(_PROBLEM, m):
        support += 2.0
    if re.search(r"(machine|maquininha|maquina|terminal)", m) and re.search(r"(error|erro|declin|negad|connect|conect|internet|offline)", m):
        support += 1.5
    if re.search(_MY_DATA, m):
        support += 1.5

    knowledge = sum(1.0 for p in _KNOWLEDGE_HINTS if re.search(p, m))
    general = bool(re.search(_GENERAL, m))
    if general:
        knowledge += 2.0

    if support == 0 and knowledge == 0:
        return RouteDecision(
            plan=[AgentName.KNOWLEDGE], intent="product_info", confidence=0.45, language=language,
            reasoning="no strong signal; default to knowledge",
        )
    if support > 0 and knowledge >= 2 and not general and re.search(r"\b(and|e|also|tambem)\b.*\b(how|como|what|o que)\b", m):
        return RouteDecision(
            plan=[AgentName.SUPPORT, AgentName.KNOWLEDGE], intent="account_data", confidence=0.7,
            language=language, reasoning="personal data plus product question",
        )
    if support >= knowledge or support >= 2:
        technical = bool(re.search(r"(machine|maquininha|maquina|terminal|error|erro|declin|negad|connect|conect)", m))
        margin = min(1.0, 0.6 + 0.1 * (support - min(knowledge, support)))
        return RouteDecision(
            plan=[AgentName.SUPPORT], intent="technical_issue" if technical else "account_data",
            confidence=round(max(0.6, margin), 2), language=language, reasoning="needs customer data",
        )
    return RouteDecision(
        plan=[AgentName.KNOWLEDGE], intent="general_info" if general else "product_info",
        confidence=round(min(0.95, 0.6 + 0.1 * knowledge), 2), language=language,
        reasoning="general or product question",
    )


class RouterAgent:
    name = AgentName.ROUTER

    def __init__(self, settings: Settings, llm: LLMClient | None) -> None:
        self.settings = settings
        self.llm = llm

    def route(self, message: str, history: list[dict[str, str]] | None = None) -> tuple[RouteDecision, str]:
        language = detect_language(message)
        with span("agent.router") as s:
            decision: RouteDecision | None = None
            mode = "offline"
            if self.llm is not None:
                try:
                    decision = self._route_llm(message, history or [], language)
                    mode = "llm"
                except (LLMError, ValueError) as exc:
                    AGENT_ERRORS.labels(self.name.value).inc()
                    log.warning("router LLM failed, using rules", extra={"error": str(exc)})
            if decision is None:
                decision = offline_route(message, language)
            s.attributes.update(mode=mode, **decision.model_dump(mode="json"))
        ROUTES.labels(decision.intent, decision.plan[0].value if decision.plan else "none").inc()
        ROUTER_CONFIDENCE.observe(decision.confidence)
        return decision, mode

    def _route_llm(self, message: str, history: list[dict[str, str]], language: str) -> RouteDecision:
        assert self.llm is not None
        convo = "\n".join(f"{h['role']}: {h['content'][:400]}" for h in history[-4:])
        content = (
            (f"<recent_conversation>\n{convo}\n</recent_conversation>\n" if convo else "")
            + f"<customer_message>\n{message}\n</customer_message>"
        )
        data: dict[str, Any] = self.llm.create_json(
            system=ROUTER_SYSTEM,
            messages=[{"role": "user", "content": content}],
            json_schema=ROUTER_SCHEMA,
            model=self.settings.router_model,
            effort=self.settings.router_effort,
            max_tokens=2000,
            name="router",
        )
        plan = [AgentName(p) for p in dict.fromkeys(data.get("plan", []))]  # dedupe, keep order
        lang = data.get("language") or language
        return RouteDecision(
            plan=plan,
            intent=data["intent"],
            confidence=max(0.0, min(1.0, float(data["confidence"]))),
            language=lang if lang in {"pt", "en", "es"} else language,
            reasoning=data.get("reasoning", ""),
            needs_human=bool(data.get("needs_human")),
        )
