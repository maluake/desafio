"""PII detection and redaction (Brazil-aware: card PANs with Luhn, CPF/CNPJ with check digits)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PAN = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_CPF = re.compile(r"(?<!\d)\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?!\d)")
_CVV = re.compile(r"\b(cvv|cvc|c[oó]digo de seguran[cç]a|security code)\b\D{0,12}(\d{3,4})\b", re.I)
_SECRET = re.compile(
    r"\b(senha|password|passwd|pin|token)\b(\s*(?:é|eh|is|:|=)\s*|\s+(?=\S*\d))([^\s,;]{3,})",
    re.I,
)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


def luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def cpf_ok(cpf: str) -> bool:
    n = [int(c) for c in cpf if c.isdigit()]
    if len(n) != 11 or len(set(n)) == 1:
        return False
    for size in (9, 10):
        s = sum(n[i] * (size + 1 - i) for i in range(size))
        dv = (s * 10) % 11 % 10
        if dv != n[size]:
            return False
    return True


@dataclass
class RedactionResult:
    text: str
    found: list[str] = field(default_factory=list)

    @property
    def redacted(self) -> bool:
        return bool(self.found)


def redact(text: str) -> RedactionResult:
    found: list[str] = []

    def _pan(m: re.Match[str]) -> str:
        if luhn_ok(m.group(0)):
            found.append("card_number")
            digits = re.sub(r"\D", "", m.group(0))
            return f"[CARD ****{digits[-4:]}]"
        return m.group(0)

    def _cpf(m: re.Match[str]) -> str:
        if cpf_ok(m.group(0)):
            found.append("cpf")
            return "[CPF REDACTED]"
        return m.group(0)

    def _cvv(m: re.Match[str]) -> str:
        found.append("cvv")
        return f"{m.group(1)} [REDACTED]"

    def _secret(m: re.Match[str]) -> str:
        found.append("secret")
        return f"{m.group(1)}{m.group(2)}[REDACTED]"

    text = _CVV.sub(_cvv, text)
    text = _SECRET.sub(_secret, text)
    text = _PAN.sub(_pan, text)
    text = _CPF.sub(_cpf, text)
    if _EMAIL.search(text):
        found.append("email")
        text = _EMAIL.sub("[EMAIL REDACTED]", text)
    return RedactionResult(text=text, found=sorted(set(found)))
