"""Customer Support Agent: resolves account-specific requests with the merchant's own data."""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from app.agents.prompts import SUPPORT_SYSTEM
from app.core.agent_base import AgentContext, BaseAgent, dedupe_sources, handoff_tool
from app.core.language import t
from app.core.tools import Tool, ToolContext, ToolExecutor
from app.data.repository import DECLINE_CODES
from app.rag.text import normalize
from app.schemas import AgentName, AgentResult, Source

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def brl(v: float) -> str:
    return "R$ " + f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def br_date(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{d}/{m}/{y}"


def _parse_date(value: str | None, today: date) -> date:
    if not value:
        return today - timedelta(days=1)
    v = normalize(value)
    if v in {"yesterday", "ontem"}:
        return today - timedelta(days=1)
    if v in {"today", "hoje"}:
        return today
    return date.fromisoformat(value)


# --------------------------------------------------------------------- tool handlers
# NOTE: every handler uses ctx.user_id (authenticated) - never a model-provided id.


def _profile(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    return ctx.services.customers.profile(ctx.user_id)


def _sales(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    repo = ctx.services.customers
    start = _parse_date(args.get("start_date"), repo.today)
    end = _parse_date(args.get("end_date") or args.get("start_date"), repo.today)
    rows = repo.sales(ctx.user_id, start, end, args.get("status"))
    approved = [r for r in rows if r["status"] == "approved"]
    return {
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "count": len(rows),
        "approved_count": len(approved),
        "approved_gross_total": round(sum(r["amount"] for r in approved), 2),
        "sales": rows,
    }


def _forecast(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    repo = ctx.services.customers
    return repo.settlement_forecast(ctx.user_id, _parse_date(args.get("sale_date"), repo.today))


def _terminals(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    terms = ctx.services.customers.terminals(ctx.user_id, args.get("serial"))
    for term in terms:
        diag = []
        if term["status"] != "online":
            diag.append(f"offline for ~{term['last_seen_minutes_ago']} min")
        if term["connection"] == "wifi" and term["wifi_signal"] in {"weak", "none"}:
            diag.append("weak Wi-Fi signal")
        if term["chip_signal"] == "none":
            diag.append("no mobile-data (chip) signal")
        if term["battery"] < 20:
            diag.append("low battery")
        if term["firmware"] != term["firmware_latest"]:
            diag.append(f"firmware outdated ({term['firmware']} -> {term['firmware_latest']})")
        term["diagnostics"] = diag or ["no issues detected"]
    return {"terminals": terms}


def _declines(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    repo = ctx.services.customers
    days = max(1, min(int(args.get("days") or 7), 90))
    rows = repo.sales(ctx.user_id, repo.today - timedelta(days=days), repo.today, status="declined")
    for r in rows:
        r["code_info"] = DECLINE_CODES.get(r.get("decline_code", ""), {"meaning": "unknown code"})
    by_terminal: dict[str, int] = {}
    for r in rows:
        by_terminal[r["terminal"]] = by_terminal.get(r["terminal"], 0) + 1
    return {"days": days, "declined": rows, "declines_by_terminal": by_terminal}


def _explain_code(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    code = str(args["code"]).strip().upper().zfill(2) if str(args["code"]).strip().isdigit() else str(args["code"]).upper()
    info = DECLINE_CODES.get(code)
    if info is None:
        return {"code": code, "known": False, "advice": "Check the machine manual or contact Getnet support."}
    return {"code": code, "known": True, **info}


def _help_center(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    hits = ctx.services.kb.search(args["query"], top_k=3)
    registry: list[dict[str, Any]] = ctx.extras.setdefault("kb_sources", [])
    passages = []
    for h in hits:
        if h.score < ctx.services.settings.retrieval_min_score:
            continue
        ref = next((r for r in registry if r["chunk_id"] == h.chunk.chunk_id), None)
        if ref is None:
            ref = {"ref": len(registry) + 1, "chunk_id": h.chunk.chunk_id, "title": h.chunk.title,
                   "section": h.chunk.section, "url": h.chunk.url, "score": round(h.score, 3)}
            registry.append(ref)
        passages.append({"ref": ref["ref"], "title": h.chunk.title, "text": h.chunk.text})
    return {"query": args["query"], "passages": passages}


def _open_ticket(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    ticket = ctx.services.tickets.create(
        ctx.user_id,
        category=args["category"],
        summary=args["summary"][:500],
        priority=args.get("priority") or "normal",
        source="support_agent",
    )
    ctx.extras["ticket_id"] = ticket["ticket_id"]
    sla = {"urgent": "2 hours", "high": "8 business hours", "normal": "2 business days"}[ticket["priority"]]
    return {"ticket_id": ticket["ticket_id"], "status": ticket["status"], "priority": ticket["priority"], "expected_response": sla}


class SupportAgent(BaseAgent):
    name = AgentName.SUPPORT

    def tools(self) -> list[Tool]:
        no_args = {"type": "object", "properties": {}}
        return [
            Tool("get_customer_profile", "Merchant profile: name, contact, plan and settlement rules, fees, masked settlement account, machines.", no_args, _profile),
            Tool(
                "get_sales",
                "List the merchant's card/Pix sales in a date range with totals. Dates: ISO YYYY-MM-DD, 'today' or 'yesterday'.",
                {
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string"},
                        "end_date": {"type": "string"},
                        "status": {"type": "string", "enum": ["approved", "declined"]},
                    },
                    "required": ["start_date"],
                },
                _sales,
            ),
            Tool(
                "get_settlement_forecast",
                "Expected deposit dates and net amounts for approved sales made on a given day, according to the merchant's plan.",
                {"type": "object", "properties": {"sale_date": {"type": "string", "description": "ISO date, 'today' or 'yesterday'"}}, "required": ["sale_date"]},
                _forecast,
            ),
            Tool(
                "get_terminal_status",
                "Live status and diagnostics of the merchant's card machines (connectivity, Wi-Fi/chip signal, battery, firmware, last seen).",
                {"type": "object", "properties": {"serial": {"type": "string", "description": "Optional serial to filter"}}},
                _terminals,
            ),
            Tool(
                "get_recent_declines",
                "Declined transactions in the last N days with response-code meanings and counts per machine.",
                {"type": "object", "properties": {"days": {"type": "integer", "description": "1-90, default 7"}}},
                _declines,
            ),
            Tool(
                "explain_decline_code",
                "Meaning, responsible party and recommended action for a card response/decline code (e.g. 05, 51, 57).",
                {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
                _explain_code,
            ),
            Tool(
                "search_help_center",
                "Search Getnet help-center guides (troubleshooting, decline codes, settlement rules). Returns numbered passages.",
                {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                _help_center,
            ),
            Tool(
                "open_support_ticket",
                "Open a support ticket for the merchant (e.g. faulty machine, replacement, settlement dispute). Returns the ticket number.",
                {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": ["terminal", "settlement", "transactions", "account", "other"]},
                        "summary": {"type": "string"},
                        "priority": {"type": "string", "enum": ["normal", "high", "urgent"]},
                    },
                    "required": ["category", "summary"],
                },
                _open_ticket,
            ),
            handoff_tool(),
        ]

    def system_prompt(self, ctx: AgentContext) -> str:
        today = date.fromisoformat(ctx.today) if ctx.today else self.services.customers.today
        return SUPPORT_SYSTEM.format(
            today=today.isoformat(),
            weekday=WEEKDAYS[today.weekday()],
            yesterday=(today - timedelta(days=1)).isoformat(),
            language="Portuguese (pt-BR)" if ctx.language == "pt" else "English" if ctx.language == "en" else ctx.language,
        )

    def finalize(self, answer: str, executor: ToolExecutor, collected: list[Any]) -> AgentResult:
        return self._result(answer, executor.ctx)

    def _result(self, answer: str, tctx: ToolContext, confidence: float = 0.9, **kw: Any) -> AgentResult:
        sources = [
            Source(title=f"[{r['ref']}] {r['title']}", url=r["url"], snippet=r["section"], score=r["score"])
            for r in tctx.extras.get("kb_sources", [])
        ]
        return AgentResult(
            agent=self.name,
            answer=answer,
            sources=dedupe_sources(sources),
            confidence=confidence,
            ticket_id=tctx.extras.get("ticket_id"),
            data={"grounded": True},
            **kw,
        )

    # ------------------------------------------------------------------ offline path
    def run_offline(self, ctx: AgentContext, executor: ToolExecutor) -> AgentResult:
        lang = ctx.language
        if not self.services.customers.exists(ctx.user_id):
            text = t(
                lang,
                "I couldn't find a Getnet merchant account linked to your login, so I can't look up your data. "
                "I'll connect you with a specialist.",
                "Não encontrei uma conta de cliente Getnet vinculada ao seu login, então não consigo consultar seus dados. "
                "Vou encaminhar você a um especialista.",
            )
            return AgentResult(agent=self.name, answer=text, confidence=0.2, needs_human=True, handoff_reason="customer_not_found")

        msg = normalize(ctx.message)
        if re.search(r"(connect|conect|internet|wi-?fi|offline|sinal|signal|chip)", msg):
            return self._offline_connectivity(ctx, executor)
        if re.search(r"(declin|negad|recusad|error|erro|codigo|code|nao autoriz|not authori)", msg):
            return self._offline_declines(ctx, executor)
        if re.search(r"(deposit|money|dinheiro|receb|cai|caiu|paid|payout|liquid|when will|quando)", msg):
            return self._offline_settlement(ctx, executor)
        if re.search(r"(sales|vendas|transac|vendi|sold)", msg):
            return self._offline_sales(ctx, executor)
        profile, _ = executor.run("get_customer_profile", {})
        text = t(
            lang,
            f"Hi {profile['contact_name']}! Your plan is {profile['plan']['name']} and you have "
            f"{len(profile['terminals'])} machine(s). How can I help with your sales, deposits or machines?",
            f"Olá, {profile['contact_name']}! Seu plano é {profile['plan']['name']} e você tem "
            f"{len(profile['terminals'])} maquininha(s). Como posso ajudar com vendas, depósitos ou maquininhas?",
        )
        return self._result(text, executor.ctx, confidence=0.6)

    def _offline_settlement(self, ctx: AgentContext, executor: ToolExecutor) -> AgentResult:
        lang = ctx.language
        msg = normalize(ctx.message)
        day = "today" if re.search(r"\b(today|hoje)\b", msg) and not re.search(r"\b(yesterday|ontem)\b", msg) else "yesterday"
        profile, _ = executor.run("get_customer_profile", {})
        fc, ok = executor.run("get_settlement_forecast", {"sale_date": day})
        if not ok:
            return self._result(t(lang, "I couldn't load your receivables right now.", "Não consegui carregar seus recebíveis agora."), executor.ctx, 0.3, needs_human=True, handoff_reason="tool_failure")
        acct = fc["settlement_account"]
        lines = []
        for dep in fc["deposits"]:
            lines.append(
                f"- {dep['sale_id']} ({dep['method']}): {brl(dep['net_amount'])} "
                + t(lang, "on ", "em ") + br_date(dep["expected_date"]) + f" - {dep['rule']}"
            )
        totals = "; ".join(f"{br_date(d)}: {brl(v)}" for d, v in fc["totals_by_date"].items())
        text = t(
            lang,
            f"Hi {profile['contact_name']}! For your approved sales on {br_date(fc['sale_date'])} "
            f"(gross {brl(fc['gross_total'])}, net {brl(fc['net_total'])}) under plan {fc['plan']}, the deposits to your "
            f"{acct['bank']} account {acct['account']} are expected as follows:\n" + "\n".join(lines)
            + f"\n\nTotals by date: {totals}.\n{fc['note']}",
            f"Olá, {profile['contact_name']}! Para suas vendas aprovadas em {br_date(fc['sale_date'])} "
            f"(bruto {brl(fc['gross_total'])}, líquido {brl(fc['net_total'])}) no plano {fc['plan']}, os depósitos na sua "
            f"conta {acct['bank']} {acct['account']} estão previstos assim:\n" + "\n".join(lines)
            + f"\n\nTotais por data: {totals}.\nValores líquidos de taxas; datas em fim de semana/feriado passam para o próximo dia útil.",
        )
        if not fc["automatic_advance"]:
            text += t(
                lang,
                "\nTip: if you need the credit amounts sooner, you can simulate a receivables advance (antecipação) in the Getnet app.",
                "\nDica: se precisar dos valores de crédito antes, você pode simular uma antecipação de recebíveis no app Getnet.",
            )
        return self._result(text, executor.ctx)

    def _offline_sales(self, ctx: AgentContext, executor: ToolExecutor) -> AgentResult:
        lang = ctx.language
        msg = normalize(ctx.message)
        day = "today" if re.search(r"\b(today|hoje)\b", msg) else "yesterday"
        out, _ = executor.run("get_sales", {"start_date": day})
        text = t(
            lang,
            f"On {br_date(out['period']['start'])} you had {out['count']} transactions, {out['approved_count']} approved, "
            f"totalling {brl(out['approved_gross_total'])} (gross).",
            f"Em {br_date(out['period']['start'])} você teve {out['count']} transações, {out['approved_count']} aprovadas, "
            f"somando {brl(out['approved_gross_total'])} (bruto).",
        )
        return self._result(text, executor.ctx)

    def _offline_connectivity(self, ctx: AgentContext, executor: ToolExecutor) -> AgentResult:
        lang = ctx.language
        status, _ = executor.run("get_terminal_status", {})
        executor.run("search_help_center", {"query": "card machine won't connect to the internet troubleshooting"})
        problem = [tm for tm in status["terminals"] if tm["status"] != "online"] or status["terminals"][:1]
        tm = problem[0]
        diag = ", ".join(tm["diagnostics"])
        steps_en = [
            "Restart the machine (hold the power button, wait 10 seconds, turn it on).",
        ]
        steps_pt = ["Reinicie a maquininha (segure o botão, aguarde 10 segundos e ligue novamente)."]
        if tm["connection"] == "wifi":
            steps_en += [
                "Move it closer to the router and re-enter the Wi-Fi password (prefer a 2.4 GHz network).",
                f"If Wi-Fi stays weak, turn Wi-Fi off so it falls back to the {tm['chip_carrier']} chip"
                + (" - but the chip currently has no signal, so check that it is well seated with the machine off." if tm["chip_signal"] == "none" else "."),
            ]
            steps_pt += [
                "Aproxime do roteador e digite novamente a senha do Wi-Fi (prefira rede 2.4 GHz).",
                f"Se o Wi-Fi continuar fraco, desligue o Wi-Fi para usar o chip {tm['chip_carrier']}"
                + (" - mas o chip está sem sinal agora; com a máquina desligada, confira se ele está bem encaixado." if tm["chip_signal"] == "none" else "."),
            ]
        else:
            steps_en.append("Check the chip is well seated (machine off) and try a spot with better carrier signal.")
            steps_pt.append("Confira se o chip está bem encaixado (com a máquina desligada) e teste em local com melhor sinal.")
        if tm["firmware"] != tm["firmware_latest"]:
            steps_en.append(f"Once online, update the firmware to {tm['firmware_latest']} (Settings > Update).")
            steps_pt.append(f"Assim que conectar, atualize o firmware para {tm['firmware_latest']} (Configurações > Atualização).")
        steps_en.append("Meanwhile, keep selling with the Payment Link or Pix in the Getnet app.")
        steps_pt.append("Enquanto isso, continue vendendo com o Link de Pagamento ou Pix pelo app Getnet.")

        ticket_txt = ""
        if tm["status"] != "online" and tm["last_seen_minutes_ago"] > 60:
            tk, ok = executor.run(
                "open_support_ticket",
                {
                    "category": "terminal",
                    "summary": f"{tm['model']} {tm['serial']} offline for {tm['last_seen_minutes_ago']} min. Diagnostics: {diag}.",
                    "priority": "high",
                },
            )
            if ok:
                ticket_txt = t(
                    lang,
                    f"\n\nBecause the machine has been offline for over an hour, I opened ticket {tk['ticket_id']} "
                    f"(expected response: {tk['expected_response']}). If the steps above solve it, no action is needed.",
                    f"\n\nComo a máquina está offline há mais de uma hora, abri o chamado {tk['ticket_id']} "
                    f"(retorno previsto: {tk['expected_response']}). Se os passos acima resolverem, não é preciso fazer nada.",
                )
        steps = steps_pt if lang == "pt" else steps_en
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        text = t(
            lang,
            f"I checked your machines: your {tm['model']} (serial ending {tm['serial'][-4:]}) is {tm['status']} - {diag}. Try this:\n",
            f"Verifiquei suas maquininhas: sua {tm['model']} (série final {tm['serial'][-4:]}) está {tm['status']} - {diag}. Tente o seguinte:\n",
        ) + numbered + ticket_txt
        return self._result(text, executor.ctx)

    def _offline_declines(self, ctx: AgentContext, executor: ToolExecutor) -> AgentResult:
        lang = ctx.language
        out, _ = executor.run("get_recent_declines", {"days": 7})
        executor.run("search_help_center", {"query": "transaction declined decline codes"})
        rows = out["declined"]
        if not rows:
            text = t(
                lang,
                "I found no declined transactions in the last 7 days. If the machine shows a code, tell me the code and I'll explain it.",
                "Não encontrei transações negadas nos últimos 7 dias. Se a máquina mostrar um código, me diga qual é e eu explico.",
            )
            return self._result(text, executor.ctx, confidence=0.7)
        lines = []
        for r in rows[-5:]:
            info = r["code_info"]
            lines.append(
                f"- {br_date(r['date'])} {r['time']} - {brl(r['amount'])} ({r.get('brand', '')}) - "
                + t(lang, "code ", "código ") + f"{r.get('decline_code')}: {info.get('meaning')}. {info.get('action', '')}"
            )
        worst = max(out["declines_by_terminal"].items(), key=lambda kv: kv[1])
        extra = ""
        if worst[1] >= 3:
            extra = t(
                lang,
                f"\n\nMachine ...{worst[0][-4:]} concentrates {worst[1]} declines; since they are different codes, also check its connection (I can run a diagnosis).",
                f"\n\nA máquina ...{worst[0][-4:]} concentra {worst[1]} recusas; como são códigos diferentes, vale verificar a conexão dela (posso fazer um diagnóstico).",
            )
        text = t(
            lang,
            "Here are your recent declined transactions and what each code means:\n",
            "Estas são suas transações negadas recentes e o significado de cada código:\n",
        ) + "\n".join(lines) + t(
            lang,
            "\n\nMost of these codes come from the shopper's bank (issuer): ask the customer to contact their bank or pay another way (Pix, another card).",
            "\n\nA maioria desses códigos vem do banco do comprador (emissor): oriente o cliente a falar com o banco ou pagar de outra forma (Pix, outro cartão).",
        ) + extra
        return self._result(text, executor.ctx)
