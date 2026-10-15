from datetime import date

import pytest

from app.config import ROOT_DIR
from app.data.repository import CustomerNotFound, CustomerRepository, TicketStore, add_business_days, next_business_day

TODAY = date(2026, 10, 15)  # Thursday


@pytest.fixture
def repo():
    return CustomerRepository(ROOT_DIR / "data" / "customers" / "customers.json", today=TODAY)


def test_business_days():
    assert add_business_days(date(2026, 10, 16), 1) == date(2026, 10, 19)  # Fri -> Mon
    assert next_business_day(date(2026, 12, 25)) == date(2026, 12, 28)  # Christmas Friday -> Monday
    assert add_business_days(date(2026, 11, 13), 1) == date(2026, 11, 16)


def test_settlement_forecast_rules(repo):
    fc = repo.settlement_forecast("cliente1988", date(2026, 10, 14))
    by_sale = {}
    for d in fc["deposits"]:
        by_sale.setdefault(d["sale_id"], []).append(d)
    assert by_sale["TX-1001"][0]["expected_date"] == "2026-10-15"  # debit D+1
    assert by_sale["TX-1003"][0]["expected_date"] == "2026-10-14"  # pix same day
    assert by_sale["TX-1002"][0]["expected_date"] == "2026-11-13"  # credit D+30 (Fri)
    assert len(by_sale["TX-1004"]) == 3  # 3 installments, monthly
    assert "TX-1006" not in by_sale  # declined sales are not settled
    assert fc["net_total"] < fc["gross_total"]


def test_automatic_advance_plan(repo):
    fc = repo.settlement_forecast("cliente2024", date(2026, 10, 14))
    inst = [d for d in fc["deposits"] if d["sale_id"] == "TX-2001"]
    assert len(inst) == 1 and inst[0]["expected_date"] == "2026-10-16"  # D+2 business days


def test_unknown_customer(repo):
    with pytest.raises(CustomerNotFound):
        repo.profile("nobody")


def test_ticket_store_persists(tmp_path):
    store = TicketStore(tmp_path)
    t = store.create("u1", "terminal", "offline", "high")
    again = TicketStore(tmp_path)
    assert again.get(t["ticket_id"])["priority"] == "high"
    assert [x["ticket_id"] for x in again.list_for("u1")] == [t["ticket_id"]]
