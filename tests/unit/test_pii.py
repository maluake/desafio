import pytest

from app.guardrails.pii import cpf_ok, luhn_ok, redact


def test_luhn():
    assert luhn_ok("4111 1111 1111 1111")
    assert not luhn_ok("4111 1111 1111 1112")
    assert not luhn_ok("1234")


def test_cpf_check_digits():
    assert cpf_ok("529.982.247-25")
    assert not cpf_ok("529.982.247-26")
    assert not cpf_ok("111.111.111-11")


@pytest.mark.parametrize(
    "text,flag,leftover",
    [
        ("my card is 4111 1111 1111 1111 ok", "card_number", "4111 1111 1111 1111"),
        ("CPF 529.982.247-25", "cpf", "529.982.247-25"),
        ("o cvv é 123", "cvv", "123"),
        ("my password is hunter22", "secret", "hunter22"),
        ("senha: abc123", "secret", "abc123"),
        ("mail me at ana@example.com", "email", "ana@example.com"),
    ],
)
def test_redaction(text, flag, leftover):
    r = redact(text)
    assert flag in r.found
    assert leftover not in r.text


def test_card_keeps_last_four():
    assert "[CARD ****1111]" in redact("4111111111111111").text


@pytest.mark.parametrize(
    "text",
    [
        "re-enter the Wi-Fi password (prefer a 2.4 GHz network)",
        "WhatsApp (11) 97694-6815",
        "order 123456789012 is late",  # 12 digits: not a PAN length / fails Luhn
        "Get Smart serial GS-7781-0042",
    ],
)
def test_no_false_positives(text):
    assert redact(text).text == text
