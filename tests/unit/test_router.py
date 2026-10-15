import pytest

from app.agents.router import offline_route
from app.schemas import AgentName as A

SCENARIOS = [
    ("What's the difference between the Get Clássica and the Get Smart?", [A.KNOWLEDGE]),
    ("What's the weather forecast in Porto Alegre tomorrow?", [A.KNOWLEDGE]),
    ("When will the money from yesterday's sales be deposited?", [A.SUPPORT]),
    ("Do I need a bank account to receive my sales via Pix?", [A.KNOWLEDGE]),
    ("My card machine won't connect to the internet, what should I do?", [A.SUPPORT]),
    ("How does receivables advance (antecipação) work with Getnet?", [A.KNOWLEDGE]),
    ("What's the euro exchange rate today?", [A.KNOWLEDGE]),
    ("My card machine is showing a transaction decline error.", [A.SUPPORT]),
    ("How many installments can I split a sale into with the crediário?", [A.KNOWLEDGE]),
    ("Can I sell through WhatsApp using the Payment Link?", [A.KNOWLEDGE]),
    ("Quando cai o dinheiro das vendas de ontem?", [A.SUPPORT]),
    ("minha maquininha está dando erro de transação negada", [A.SUPPORT]),
    ("I want to talk to a human", [A.ESCALATION]),
    ("hi!", [A.KNOWLEDGE]),
    ("write me a poem about cats", []),
]


@pytest.mark.parametrize("message,plan", SCENARIOS)
def test_offline_router_plans(message, plan):
    assert offline_route(message, "en").plan == plan


def test_mixed_request_runs_two_agents():
    d = offline_route("When will yesterday's sales be deposited and how does antecipação work?", "en")
    assert d.plan == [A.SUPPORT, A.KNOWLEDGE]


def test_vague_message_has_low_confidence():
    assert offline_route("hmm", "en").confidence < 0.55
