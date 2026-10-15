"""Orchestrator: the message workflow that connects guardrails, router and specialists.

    message
      -> input guardrails (validate, redact PII, injection/unsafe/cross-account checks)
      -> Router Agent (plan = ordered list of specialists + intent + confidence)
      -> specialists run sequentially; each receives the previous agents' results
         (direct in-process calls with a typed AgentContext/AgentResult contract)
      -> escalation decision (explicit request, sensitive topic, low confidence, agent handoff)
      -> composition (single answer, or LLM merge of several specialist answers)
      -> output guardrails (PII leak, prompt leak, grounding) -> memory -> metrics -> response
"""

from __future__ import annotations

import time
from datetime import date

from app.agents.escalation import EscalationAgent
from app.agents.knowledge import KnowledgeAgent
from app.agents.prompts import COMPOSER_SYSTEM
from app.agents.router import RouterAgent
from app.agents.support import SupportAgent
from app.container import Services
from app.core.agent_base import AgentContext, dedupe_sources
from app.core.language import detect_language, t
from app.core.llm import LLMError, text_of
from app.core.observability import GUARDRAIL_EVENTS, REQUEST_LATENCY, REQUESTS, get_logger, span, start_trace
from app.guardrails.policies import PII_WARNING, block_message, check_input, check_output
from app.schemas import AgentName, AgentResult, ChatResponse, GuardrailReport, RouteDecision

log = get_logger(__name__)


class Orchestrator:
    def __init__(self, services: Services) -> None:
        self.s = services
        self.settings = services.settings
        llm = services.llm
        self.router = RouterAgent(self.settings, llm)
        self.knowledge = KnowledgeAgent(self.settings, llm, services)
        self.support = SupportAgent(self.settings, llm, services)
        self.escalation = EscalationAgent(self.settings, llm, services)
        self.specialists = {AgentName.KNOWLEDGE: self.knowledge, AgentName.SUPPORT: self.support}

    # ------------------------------------------------------------------ public
    def handle(self, message: str, user_id: str) -> ChatResponse:
        trace = start_trace()
        started = time.perf_counter()
        with span("orchestrator.handle", user_id=user_id):
            response = self._handle(message, user_id, trace.trace_id)
        response.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        if self.settings.expose_trace:
            response.trace = trace.to_list()
        REQUESTS.labels(response.agent.value, response.mode, str(response.escalated).lower()).inc()
        REQUEST_LATENCY.observe(response.latency_ms / 1000)
        log.info(
            "chat handled",
            extra={
                "user_id": user_id,
                "final_agent": response.agent.value,
                "intent": response.route.intent if response.route else None,
                "plan": [a.value for a in response.route.plan] if response.route else [],
                "confidence": response.route.confidence if response.route else None,
                "tools": [c.tool for c in response.tool_calls],
                "escalated": response.escalated,
                "guardrails": response.guardrails.model_dump(),
                "latency_ms": response.latency_ms,
                "mode": response.mode,
            },
        )
        return response

    # ------------------------------------------------------------------ workflow
    def _handle(self, message: str, user_id: str, trace_id: str) -> ChatResponse:
        lang = detect_language(message)
        with span("guardrails.input") as gs:
            verdict = check_input(message, user_id, self.settings.max_message_chars)
            gs.attributes.update(flags=verdict.flags, blocked=verdict.blocked)
        for f in verdict.flags:
            GUARDRAIL_EVENTS.labels("input", f.split(":")[0] if f.startswith("pii") else f).inc()
        report = GuardrailReport(input_flags=verdict.flags, pii_redacted=verdict.pii_redacted, blocked=verdict.blocked)

        if verdict.blocked:
            return ChatResponse(
                trace_id=trace_id, user_id=user_id, response=block_message(verdict.block_reason or "", lang),
                agent=AgentName.GUARDRAIL, guardrails=report, mode="rules",
            )

        history = self.s.memory.get(user_id)
        route, router_mode = self.router.route(verdict.text, history)
        lang = route.language if route.language in {"pt", "en"} else lang
        route = self._apply_policy(route, verdict.escalate_reason)

        if route.intent == "unsupported" or not route.plan:
            text = t(
                lang,
                "Sorry, that's outside what I can help with. I can answer questions about Getnet products, your sales "
                "and receivables, card machine issues, and quick lookups like weather or exchange rates.",
                "Desculpe, isso está fora do que posso ajudar. Posso responder sobre produtos Getnet, suas vendas e "
                "recebíveis, problemas na maquininha e consultas rápidas como clima ou câmbio.",
            )
            return ChatResponse(trace_id=trace_id, user_id=user_id, response=text, agent=AgentName.ROUTER,
                                route=route, guardrails=report, mode=router_mode)

        ctx = AgentContext(
            message=verdict.text, user_id=user_id, language=lang, history=history, route=route,
            today=self.s.customers.today.isoformat() if isinstance(self.s.customers.today, date) else None,
        )
        results: list[AgentResult] = []
        for agent_name in route.plan:
            if agent_name == AgentName.ESCALATION:
                continue  # escalation is decided after the specialists ran
            agent = self.specialists[agent_name]
            ctx.prior_results = list(results)
            results.append(agent.run(ctx))

        escalation_reason = self._escalation_reason(route, verdict.escalate_reason, results)
        if escalation_reason:
            ctx.prior_results = list(results)
            results.append(self.escalation.run(ctx, escalation_reason))

        answer = self._compose(results, lang)
        grounded = False if any(r.data.get("grounded") is False for r in results) else None
        with span("guardrails.output") as gs:
            out = check_output(answer, grounded=grounded, lang=lang)
            gs.attributes["flags"] = out.flags
        for f in out.flags:
            GUARDRAIL_EVENTS.labels("output", f).inc()
        report.output_flags = out.flags
        final_text = out.text
        if verdict.pii_redacted and any(f in verdict.flags for f in ("pii:card_number", "pii:cvv", "pii:secret")):
            final_text = t(lang, *PII_WARNING) + "\n\n" + final_text

        self.s.memory.append(user_id, verdict.text, final_text)
        modes = {r.data.get("mode", "llm") for r in results}
        ticket = next((r.ticket_id for r in reversed(results) if r.ticket_id), None)
        return ChatResponse(
            trace_id=trace_id,
            user_id=user_id,
            response=final_text,
            agent=results[-1].agent,
            route=route,
            agents_involved=[AgentName.ROUTER, *[r.agent for r in results]],
            sources=dedupe_sources([s for r in results for s in r.sources]),
            tool_calls=[c for r in results for c in r.tool_calls],
            escalated=any(r.agent == AgentName.ESCALATION for r in results),
            ticket_id=ticket,
            guardrails=report,
            mode="llm" if modes == {"llm"} else "offline" if modes <= {"offline"} else "mixed",
        )

    def _apply_policy(self, route: RouteDecision, escalate_reason: str | None) -> RouteDecision:
        """Deterministic business rules on top of the router's decision."""
        plan = list(route.plan)
        if escalate_reason == "human_request":
            plan = [AgentName.ESCALATION]
            route = route.model_copy(update={"intent": "human_handoff", "needs_human": True})
        elif escalate_reason in {"account_compromise", "legal_threat"} and AgentName.ESCALATION not in plan:
            plan.append(AgentName.ESCALATION)
            route = route.model_copy(update={"needs_human": True})
        return route.model_copy(update={"plan": plan})

    def _escalation_reason(self, route: RouteDecision, sensitive: str | None, results: list[AgentResult]) -> str | None:
        if sensitive:
            return sensitive
        if route.needs_human or AgentName.ESCALATION in route.plan:
            return "human_request" if route.intent == "human_handoff" else "router_requested"
        for r in results:
            if r.needs_human:
                return r.handoff_reason or "agent_requested"
        best = max((r.confidence for r in results), default=0.0)
        if route.confidence < self.settings.router_confidence_threshold and best < 0.5:
            return "low_confidence"
        return None

    def _compose(self, results: list[AgentResult], lang: str) -> str:
        specialists = [r for r in results if r.agent != AgentName.ESCALATION]
        handoff = [r for r in results if r.agent == AgentName.ESCALATION]
        if len(specialists) <= 1:
            body = specialists[0].answer if specialists else ""
        else:
            body = self._merge(specialists, lang)
        if handoff:
            body = f"{body}\n\n{handoff[0].answer}" if body else handoff[0].answer
        return body

    def _merge(self, results: list[AgentResult], lang: str) -> str:
        if self.s.llm is not None:
            try:
                with span("compose"):
                    blocks = "\n\n".join(f"<answer>\n{r.answer}\n</answer>" for r in results)
                    resp = self.s.llm.create(
                        system=COMPOSER_SYSTEM.format(language="Portuguese (pt-BR)" if lang == "pt" else "English"),
                        messages=[{"role": "user", "content": blocks}],
                        effort="low",
                        max_tokens=4000,
                        name="compose",
                    )
                    merged = text_of(resp)
                    if merged:
                        return merged
            except LLMError as exc:
                log.warning("compose failed, concatenating", extra={"error": str(exc)})
        return "\n\n".join(r.answer for r in results)
