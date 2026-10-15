"""Customer data access layer (mock of the merchant core systems).

In production this would wrap internal APIs (merchant registry, transactions ledger,
receivables calendar, terminal management/TMS, ticketing). Here it is backed by a JSON
fixture whose sale dates are *relative* to an injectable "today", so demos and tests
always have data for "yesterday".
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

# Fixed-date Brazilian national holidays (movable ones such as Carnival are omitted for brevity).
_FIXED_HOLIDAYS = {(1, 1), (4, 21), (5, 1), (9, 7), (10, 12), (11, 2), (11, 15), (11, 20), (12, 25)}


def is_business_day(d: date) -> bool:
    return d.weekday() < 5 and (d.month, d.day) not in _FIXED_HOLIDAYS


def next_business_day(d: date) -> date:
    while not is_business_day(d):
        d += timedelta(days=1)
    return d


def add_business_days(d: date, n: int) -> date:
    while n > 0:
        d += timedelta(days=1)
        if is_business_day(d):
            n -= 1
    return d


class CustomerNotFound(LookupError):
    pass


class CustomerRepository:
    def __init__(self, path: Path, today: date | None = None) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._customers = {c["user_id"]: c for c in raw["customers"]}
        self._today = today

    @property
    def today(self) -> date:
        return self._today or date.today()

    def exists(self, user_id: str) -> bool:
        return user_id in self._customers

    def _get(self, user_id: str) -> dict[str, Any]:
        try:
            return self._customers[user_id]
        except KeyError as exc:
            raise CustomerNotFound(f"no merchant registered for user_id '{user_id}'") from exc

    # ------------------------------------------------------------------ profile
    def profile(self, user_id: str) -> dict[str, Any]:
        c = self._get(user_id)
        return {
            "merchant_name": c["name"],
            "contact_name": c["contact_name"],
            "segment": c["segment"],
            "city": c["city"],
            "customer_since": c["since"],
            "status": c["status"],
            "settlement_account": c["settlement_account"],
            "plan": {k: v for k, v in c["plan"].items()},
            "terminals": [{"serial": t["serial"], "model": t["model"], "status": t["status"]} for t in c["terminals"]],
        }

    # ------------------------------------------------------------------ sales
    def sales(self, user_id: str, start: date, end: date, status: str | None = None) -> list[dict[str, Any]]:
        c = self._get(user_id)
        out = []
        for s in c["sales"]:
            d = self.today - timedelta(days=s["days_ago"])
            if start <= d <= end and (status is None or s["status"] == status):
                row = {k: v for k, v in s.items() if k != "days_ago"}
                row["date"] = d.isoformat()
                out.append(row)
        return sorted(out, key=lambda r: (r["date"], r["time"]))

    # ------------------------------------------------------------------ settlement
    def settlement_forecast(self, user_id: str, sale_date: date) -> dict[str, Any]:
        """Expected deposits for approved sales made on ``sale_date`` under the merchant's plan."""
        c = self._get(user_id)
        plan = c["plan"]
        fees = plan["fees"]
        deposits: list[dict[str, Any]] = []
        for s in self.sales(user_id, sale_date, sale_date, status="approved"):
            amount = float(s["amount"])
            method = s["method"]
            if method == "pix":
                deposits.append(self._deposit(s, sale_date, amount, fees["pix"], "Pix: same day (instant)"))
            elif method == "debit":
                when = add_business_days(sale_date, plan["debit_days"])
                deposits.append(self._deposit(s, when, amount, fees["debit"], f"Debit: D+{plan['debit_days']} business day"))
            else:
                n = int(s.get("installments", 1))
                if n == 1:
                    days = plan["credit_days"]
                    when = add_business_days(sale_date, days) if days <= 2 else next_business_day(sale_date + timedelta(days=days))
                    deposits.append(self._deposit(s, when, amount, fees["credit"], f"Credit à vista: D+{days}"))
                elif plan["installment_mode"] == "advance_d2":
                    when = add_business_days(sale_date, 2)
                    deposits.append(
                        self._deposit(s, when, amount, fees["credit_installments"], f"Credit {n}x with automatic advance: D+2")
                    )
                else:
                    per = round(amount / n, 2)
                    for k in range(1, n + 1):
                        when = next_business_day(sale_date + timedelta(days=30 * k))
                        deposits.append(
                            self._deposit(
                                s, when, per, fees["credit_installments"], f"Credit {n}x: installment {k}/{n} every 30 days"
                            )
                        )
        by_date: dict[str, float] = {}
        for d in deposits:
            by_date[d["expected_date"]] = round(by_date.get(d["expected_date"], 0.0) + d["net_amount"], 2)
        return {
            "sale_date": sale_date.isoformat(),
            "plan": plan["name"],
            "automatic_advance": plan["automatic_advance"],
            "settlement_account": c["settlement_account"],
            "deposits": deposits,
            "totals_by_date": dict(sorted(by_date.items())),
            "gross_total": round(sum(float(s["amount"]) for s in self.sales(user_id, sale_date, sale_date, "approved")), 2),
            "net_total": round(sum(d["net_amount"] for d in deposits), 2),
            "note": "Amounts are net of MDR fees. Deposits falling on weekends/holidays move to the next business day.",
        }

    @staticmethod
    def _deposit(sale: dict[str, Any], when: date, gross: float, fee: float, rule: str) -> dict[str, Any]:
        return {
            "sale_id": sale["id"],
            "method": sale["method"],
            "gross_amount": round(gross, 2),
            "fee_rate": fee,
            "net_amount": round(gross * (1 - fee), 2),
            "expected_date": when.isoformat(),
            "rule": rule,
        }

    # ------------------------------------------------------------------ terminals
    def terminals(self, user_id: str, serial: str | None = None) -> list[dict[str, Any]]:
        c = self._get(user_id)
        terms = c["terminals"]
        if serial:
            terms = [t for t in terms if t["serial"].lower() == serial.lower()]
        return [dict(t) for t in terms]


# Market-standard (ABECS) response codes used by the decline diagnosis tool.
DECLINE_CODES: dict[str, dict[str, str]] = {
    "03": {"meaning": "Installments not allowed for this card/plan", "who": "merchant/issuer", "action": "Reduce the number of installments or use credit à vista."},
    "05": {"meaning": "Not authorised by the issuing bank (generic decline)", "who": "issuer", "action": "Ask the shopper to contact their bank or use another card/Pix."},
    "08": {"meaning": "Timeout - no response in time", "who": "network", "action": "Check the machine connection; confirm in the sales report before retrying."},
    "11": {"meaning": "Installment plan not supported by the merchant's machine/plan", "who": "merchant", "action": "Reduce installments or contact Getnet to enable the plan."},
    "14": {"meaning": "Invalid card number or brand not accepted", "who": "card", "action": "Try another card."},
    "26": {"meaning": "Duplicate transaction", "who": "merchant", "action": "Check the sales report; the sale may already be approved."},
    "51": {"meaning": "Insufficient funds or credit limit exceeded", "who": "issuer", "action": "Ask the shopper to use another card or payment method."},
    "54": {"meaning": "Expired card", "who": "card", "action": "Ask for another card."},
    "55": {"meaning": "Wrong PIN", "who": "shopper", "action": "Retry carefully; repeated errors block the card at the issuer."},
    "57": {"meaning": "Transaction not permitted for this card (function not enabled)", "who": "issuer", "action": "Try the other function (debit/credit) or another card."},
    "59": {"meaning": "Suspected fraud - blocked by issuer", "who": "issuer", "action": "Shopper must contact their bank."},
    "AA": {"meaning": "Timeout at the acquirer", "who": "network", "action": "Wait a few minutes and retry after checking the sales report."},
}


class TicketStore:
    """Support/handoff tickets. Append-only JSONL so tickets survive restarts in Docker volumes."""

    def __init__(self, runtime_dir: Path) -> None:
        self.path = runtime_dir / "tickets.jsonl"
        self._lock = threading.Lock()
        self._tickets: dict[str, dict[str, Any]] = {}
        runtime_dir.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    t = json.loads(line)
                    self._tickets[t["ticket_id"]] = t

    def create(self, user_id: str, category: str, summary: str, priority: str = "normal", **extra: Any) -> dict[str, Any]:
        ticket = {
            "ticket_id": f"GN-{uuid.uuid4().hex[:8].upper()}",
            "user_id": user_id,
            "category": category,
            "summary": summary,
            "priority": priority,
            "status": "open",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            **extra,
        }
        with self._lock:
            self._tickets[ticket["ticket_id"]] = ticket
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ticket, ensure_ascii=False) + "\n")
        return ticket

    def list_for(self, user_id: str) -> list[dict[str, Any]]:
        return [t for t in self._tickets.values() if t["user_id"] == user_id]

    def get(self, ticket_id: str) -> dict[str, Any] | None:
        return self._tickets.get(ticket_id)
