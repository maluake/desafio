"""Human Escalation Agent (bonus): hands the case to a human operator.

* Writes a concise case summary + priority (LLM structured output, or rules offline).
* Creates a hand-off ticket carrying the conversation context, so the human does not
  have to ask the customer to repeat themselves.
* Notifies the operations channel (Slack / Microsoft Teams incoming webhook) if configured.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from app.agents.prompts import ESCALATION_SCHEMA, ESCALATION_SUMMARY_SYSTEM
from app.config import Settings
from app.core.agent_base import AgentContext
from app.core.language import t
from app.core.llm import LLMClient, LLMError
from app.core.observability import AGENT_LATENCY, ESCALATIONS, get_logger, span
from app.schemas import AgentName, AgentResult, ToolCallRecord

log = get_logger(__name__)

SLA = {"urgent": "2 hours", "high": "8 business hours", "normal": "2 business days"}
SLA_PT = {"urgent": "2 horas", "high": "8 horas úteis", "normal": "2 dias úteis"}

_RULES = {
    "account_compromise": ("urgent", "fraud_security"),
    "fraud_enablement": ("urgent", "fraud_security"),
    "legal_threat": ("high", "legal"),
    "customer_not_found": ("normal", "general"),
    "human_request": ("normal", "general"),
}


class EscalationAgent:
    name = AgentName.ESCALATION

    def __init__(self, settings: Settings, llm: LLMClient | None, services: Any) -> None:
        self.settings = settings
        self.llm = llm
        self.services = services

    def run(self, ctx: AgentContext, reason: str) -> AgentResult:
        start = time.perf_counter()
        with span("agent.escalation", reason=reason) as s:
            summary, priority, category, mode = self._summarise(ctx, reason)
            t0 = time.perf_counter()
            ticket = self.services.tickets.create(
                ctx.user_id,
                category=category,
                summary=summary,
                priority=priority,
                source="escalation_agent",
                reason=reason,
                transcript=[*ctx.history[-6:], {"role": "user", "content": ctx.message}],
                agent_notes=[{"agent": r.agent.value, "answer": r.answer[:600]} for r in ctx.prior_results],
            )
            records = [
                ToolCallRecord(
                    agent=self.name, tool="create_handoff_ticket",
                    arguments={"category": category, "priority": priority},
                    latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                )
            ]
            notified = self._notify(ticket)
            if notified is not None:
                records.append(ToolCallRecord(agent=self.name, tool="notify_operations_channel", arguments={}, ok=notified))
            s.attributes.update(ticket_id=ticket["ticket_id"], priority=priority, mode=mode)

        ESCALATIONS.labels(reason).inc()
        AGENT_LATENCY.labels(self.name.value).observe(time.perf_counter() - start)
        lang = ctx.language
        answer = t(
            lang,
            f"I've handed your case to a Getnet specialist (ticket {ticket['ticket_id']}, priority {priority}). "
            f"You'll be contacted within {SLA[priority]}, and they'll already have the context of this conversation.",
            f"Encaminhei seu caso para um especialista da Getnet (chamado {ticket['ticket_id']}, prioridade {priority}). "
            f"Você será contatado(a) em até {SLA_PT[priority]}, e o especialista já terá o contexto desta conversa.",
        )
        if category == "fraud_security":
            answer += t(
                lang,
                " Meanwhile, do not share passwords or codes with anyone, and if you suspect unauthorised access, "
                "change your Getnet app password now.",
                " Enquanto isso, não compartilhe senhas ou códigos com ninguém e, se suspeitar de acesso indevido, "
                "troque agora a senha do app Getnet.",
            )
        return AgentResult(
            agent=self.name, answer=answer, tool_calls=records, ticket_id=ticket["ticket_id"],
            data={"priority": priority, "category": category, "mode": mode, "grounded": True},
        )

    def _summarise(self, ctx: AgentContext, reason: str) -> tuple[str, str, str, str]:
        if self.llm is not None:
            notes = "\n".join(f"- {r.agent.value}: {r.answer[:400]}" for r in ctx.prior_results) or "- none"
            try:
                data = self.llm.create_json(
                    system=ESCALATION_SUMMARY_SYSTEM,
                    messages=[{
                        "role": "user",
                        "content": f"user_id: {ctx.user_id}\nreason: {reason}\n<customer_message>\n{ctx.message}\n"
                                   f"</customer_message>\nagent notes:\n{notes}",
                    }],
                    json_schema=ESCALATION_SCHEMA,
                    effort="low",
                    max_tokens=1500,
                    name="escalation",
                )
                return data["summary"], data["priority"], data["category"], "llm"
            except LLMError as exc:
                log.warning("escalation summary via LLM failed", extra={"error": str(exc)})
        priority, category = _RULES.get(reason, ("normal", "general"))
        if reason in {"tool_failure", "low_confidence", "no_answer_found"}:
            category = "general"
        summary = f"[{reason}] Customer {ctx.user_id} wrote: \"{ctx.message[:300]}\"."
        if ctx.prior_results:
            summary += " Automated agents tried: " + ", ".join(r.agent.value for r in ctx.prior_results) + "."
        return summary, priority, category, "offline"

    def _notify(self, ticket: dict[str, Any]) -> bool | None:
        url = self.settings.escalation_webhook_url
        if not url:
            return None
        text = f":rotating_light: New hand-off {ticket['ticket_id']} [{ticket['priority']}] {ticket['category']}: {ticket['summary']}"
        try:
            # Slack and Teams incoming webhooks both accept a simple {"text": ...} payload.
            httpx.post(url, json={"text": text}, timeout=self.settings.http_timeout_s).raise_for_status()
            return True
        except httpx.HTTPError as exc:
            log.warning("escalation webhook failed", extra={"error": str(exc)})
            return False
