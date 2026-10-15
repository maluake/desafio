"""Input and output guardrails.

Layered defence, cheapest first:

1. **Input validation** - size limits, empty input.
2. **PII redaction** - card numbers / CVV / passwords / CPF never reach the LLM, logs or memory.
3. **Prompt-injection heuristics** - known jailbreak phrasings are blocked before routing.
4. **Unsafe / disallowed content** - fraud enablement, violence, self-harm (with a supportive
   response and the CVV 188 helpline), hate.
5. **Cross-account access** - requests about another merchant's data are refused (tools are
   also bound to the authenticated user_id, so this is defence in depth).
6. **Sensitive-but-legit topics** (fraud on own account, legal threats, account hacked) are
   *not* blocked: they are flagged so the orchestrator hands off to a human.

Output: PII leakage scan, prompt-leak scan, and a grounding check for knowledge answers.
These rules are deliberately transparent and unit-tested; an LLM-based classifier can be
plugged in as an extra layer (see README "Guardrails").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.guardrails.pii import redact
from app.rag.text import normalize

INJECTION_PATTERNS = [
    r"ignore (all |the |any )?(previous|prior|above|earlier) (instructions|prompts|rules)",
    r"ignor[ea] (todas )?(as )?(instruc[oõ]es|regras) (anteriores|acima)",
    r"disregard (your|the|all) (instructions|rules|guidelines)",
    r"(reveal|show|print|repeat|output) (me )?(your|the) (system|hidden|initial) (prompt|instructions)",
    r"(mostre|revele|imprima) (o|seu) prompt( do sistema)?",
    r"\byou are now\b",
    r"\bvoc[eê] agora [eé]\b",
    r"\b(dan|developer) mode\b",
    r"\bjailbreak\b",
    r"act as (an? )?(unrestricted|unfiltered)",
    r"</?(system|customer_message|context_from_other_agents)>",
]

UNSAFE_PATTERNS: dict[str, list[str]] = {
    "fraud_enablement": [
        r"(clone|clonar|skim|copiar) (a |um |o )?(card|cart[aã]o)",
        r"(stolen|roubad[oa]s?) (card|cart[aã]o|cart[oõ]es)",
        r"(cards?|cart[aã]o|cart[oõ]es) (stolen|roubad[oa]s?|clonad[oa]s?)",
        r"(burlar|bypass|contornar|enganar) (o )?(antifraude|anti-fraud|fraud detection)",
        r"(fake|falso|falsa) (chargeback|estorno)",
        r"lavar dinheiro|money laundering|launder money",
        r"(testar|test) (cart[oõ]es|cards) (gerados|generated)",
    ],
    "violence": [r"\b(make|build|fazer|fabricar) (a |uma )?(bomb|bomba|explosive|explosivo)\b", r"\b(kill|matar) (someone|algu[eé]m)\b"],
    "self_harm": [r"\b(suicid|kill myself|me matar|tirar minha vida|end my life|self[- ]harm)"],
    "hate": [r"\b(nazi|nazista)\b.*\b(propaganda|apologia)\b"],
}

SENSITIVE_PATTERNS: dict[str, list[str]] = {
    "account_compromise": [r"(hack|invadid|invadiram|hackead|golpe|scam|phishing|fraud|fraude|clonad)"],
    "legal_threat": [r"\b(procon|processar|lawsuit|sue you|advogado|lawyer|reclame aqui|justi[cç]a)\b"],
    "human_request": [
        r"\b(human|humano|atendente|pessoa real|real person|agent|operator|operador|gerente|manager)\b.*\b(talk|speak|falar|quero|want|need|preciso)\b",
        r"\b(talk|speak|falar|quero|want|need|preciso)\b.*\b(human|humano|atendente|pessoa real|real person|operator|operador)\b",
    ],
}

_OTHER_ACCOUNT = re.compile(r"\b(?:cliente|client|user|usu[aá]rio|merchant)[ _-]?(\d{3,})\b", re.I)


@dataclass
class InputVerdict:
    text: str  # redacted text to be used downstream
    flags: list[str] = field(default_factory=list)
    blocked: bool = False
    block_reason: str | None = None
    pii_redacted: bool = False
    escalate_reason: str | None = None


def _any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)


def check_input(message: str, user_id: str, max_chars: int) -> InputVerdict:
    if len(message) > max_chars:
        return InputVerdict(text=message[:max_chars], flags=["too_long"], blocked=True, block_reason="too_long")

    red = redact(message)
    verdict = InputVerdict(text=red.text, pii_redacted=red.redacted)
    verdict.flags.extend(f"pii:{f}" for f in red.found)
    norm = normalize(message)

    if _any(INJECTION_PATTERNS, norm) or _any(INJECTION_PATTERNS, message):
        verdict.flags.append("prompt_injection")
        verdict.blocked, verdict.block_reason = True, "prompt_injection"
        return verdict

    for category, patterns in UNSAFE_PATTERNS.items():
        if _any(patterns, norm) or _any(patterns, message):
            verdict.flags.append(f"unsafe:{category}")
            verdict.blocked, verdict.block_reason = True, category
            return verdict

    for m in _OTHER_ACCOUNT.finditer(message):
        mentioned = m.group(0).replace(" ", "").replace("-", "").replace("_", "").lower()
        if mentioned != user_id.replace("_", "").replace("-", "").lower():
            verdict.flags.append("cross_account")
            verdict.blocked, verdict.block_reason = True, "cross_account"
            return verdict

    for category, patterns in SENSITIVE_PATTERNS.items():
        if _any(patterns, norm):
            verdict.flags.append(f"sensitive:{category}")
            verdict.escalate_reason = verdict.escalate_reason or category
    return verdict


BLOCK_MESSAGES = {
    "too_long": (
        "Your message is too long. Please summarise your question in a few sentences.",
        "Sua mensagem é muito longa. Por favor, resuma sua dúvida em poucas frases.",
    ),
    "prompt_injection": (
        "I can't follow that request. I'm Getnet's virtual assistant and can help with Getnet products, "
        "your sales, receivables and card machines.",
        "Não posso seguir essa solicitação. Sou o assistente virtual da Getnet e posso ajudar com produtos Getnet, "
        "suas vendas, recebíveis e maquininhas.",
    ),
    "fraud_enablement": (
        "I can't help with that. If you suspect fraud involving your business, I can connect you with Getnet's security team.",
        "Não posso ajudar com isso. Se você suspeita de fraude envolvendo seu negócio, posso conectar você à equipe de segurança da Getnet.",
    ),
    "violence": ("I can't help with that request.", "Não posso ajudar com essa solicitação."),
    "hate": ("I can't help with that request.", "Não posso ajudar com essa solicitação."),
    "self_harm": (
        "I'm really sorry you're going through this. You don't have to face it alone - in Brazil you can talk to "
        "CVV for free, 24h, by calling 188 or at cvv.org.br. If you're in immediate danger, call 192 (SAMU).",
        "Sinto muito que você esteja passando por isso. Você não precisa enfrentar isso sozinho(a) - fale com o CVV "
        "gratuitamente, 24h, pelo telefone 188 ou em cvv.org.br. Em caso de perigo imediato, ligue 192 (SAMU).",
    ),
    "cross_account": (
        "For security reasons I can only access data from the account you are logged in with.",
        "Por segurança, só posso acessar dados da conta com a qual você está autenticado(a).",
    ),
}


def block_message(reason: str, lang: str) -> str:
    en, pt = BLOCK_MESSAGES.get(reason, BLOCK_MESSAGES["violence"])
    return pt if lang == "pt" else en


PII_WARNING = (
    "For your security, please never share full card numbers, CVV or passwords in chat - I removed them from your message.",
    "Por segurança, nunca compartilhe número completo de cartão, CVV ou senhas no chat - eu removi esses dados da sua mensagem.",
)

# --------------------------------------------------------------------------- output

_LEAK_MARKERS = re.compile(r"</?(customer_message|context_from_other_agents|system)>|You are Getnet's", re.I)


@dataclass
class OutputVerdict:
    text: str
    flags: list[str] = field(default_factory=list)


def check_output(text: str, *, grounded: bool | None, lang: str) -> OutputVerdict:
    flags: list[str] = []
    red = redact(text)
    # E-mail addresses in answers are typically official contact channels - keep them.
    leaked = [f for f in red.found if f != "email"]
    if leaked:
        flags.extend(f"output_pii:{f}" for f in leaked)
        text = redact_no_email(text)
    if _LEAK_MARKERS.search(text):
        flags.append("prompt_leak")
        text = _LEAK_MARKERS.sub("", text)
    if grounded is False:
        flags.append("ungrounded")
        text += (
            "\n\n_Não encontrei essa informação na base oficial da Getnet; confirme no site getnet.net ou no app._"
            if lang == "pt"
            else "\n\n_I couldn't find this in Getnet's official knowledge base; please confirm on getnet.net or in the Getnet app._"
        )
    if not text.strip():
        flags.append("empty_output")
    return OutputVerdict(text=text.strip(), flags=flags)


def redact_no_email(text: str) -> str:
    emails = re.findall(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", text)
    out = redact(text).text
    for e in emails:
        out = out.replace("[EMAIL REDACTED]", e, 1)
    return out
