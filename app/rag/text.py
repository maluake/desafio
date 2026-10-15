"""Text normalisation, tokenisation and bilingual (EN<->PT-BR) query expansion."""

from __future__ import annotations

import re
import unicodedata

_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?")

STOPWORDS = frozenset(
    """
    a an and are as at be by can could do does for from had has have how i if in into is it its
    me my of on or our so than that the their them then there these they this to was we were
    what when where which who why will with would you your yours about any also just more most
    not no o os as um uma uns umas de da do das dos e em no na nos nas por para com sem que se
    ao aos eu meu minha meus minhas voce voces ele ela eles elas isso isto esse essa este esta
    qual quais como quando onde porque pra pro ser estar ter foi sao seu sua seus suas mais
    muito tem
    """.split()
)

# Domain glossary used to bridge English questions and Portuguese content (and vice-versa).
# Lexical retrievers do not translate; this keeps recall high without a multilingual model.
GLOSSARY: dict[str, list[str]] = {
    "receivables advance": ["antecipacao de recebiveis", "antecipacao", "receba ja"],
    "antecipacao": ["receivables advance", "advance", "anticipation"],
    "card machine": ["maquininha", "maquina de cartao", "pos", "terminal"],
    "machine": ["maquininha", "terminal"],
    "maquininha": ["card machine", "pos", "terminal"],
    "installments": ["parcelas", "parcelamento", "parcelado"],
    "installment": ["parcelas", "parcelamento"],
    "parcelas": ["installments"],
    "split": ["parcelar", "parcelamento", "installments"],
    "payment link": ["link de pagamento", "get pay"],
    "link de pagamento": ["payment link"],
    "bank account": ["conta bancaria", "domicilio bancario", "conta corrente", "settlement account"],
    "deposit": ["deposito", "recebimento", "liquidacao", "settlement", "credited"],
    "deposited": ["deposito", "recebimento", "liquidacao", "settlement", "credited"],
    "money": ["recebimento", "deposito", "valor"],
    "fees": ["taxas", "tarifas", "mdr"],
    "fee": ["taxa", "tarifa", "mdr"],
    "rent": ["aluguel", "mensalidade"],
    "sales": ["vendas", "transacoes"],
    "sale": ["venda", "transacao"],
    "decline": ["negada", "recusada", "nao autorizada", "codigo de erro"],
    "declined": ["negada", "recusada", "nao autorizada"],
    "error": ["erro", "codigo"],
    "internet": ["conexao", "wi-fi", "wifi", "chip", "3g", "4g", "sinal"],
    "connect": ["conexao", "conectar", "wi-fi", "chip", "sinal"],
    "digital account": ["conta digital"],
    "crediario": ["crediario getnet", "parcelamento ate 48x", "installments"],
    "whatsapp": ["whatsapp pay"],
    "support": ["atendimento", "central de atendimento", "service center", "suporte"],
    "contact": ["contato", "atendimento", "telefone", "phone"],
    "phone": ["telefone", "contato", "number"],
    "receipt": ["comprovante", "recibo"],
    "debit": ["debito"],
    "credit": ["credito"],
    "difference": ["comparacao", "diferenca", "vs"],
}


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize(text: str) -> str:
    return strip_accents(text.lower())


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = _WORD.findall(normalize(text))
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def expand_query(query: str) -> str:
    """Append glossary translations for every glossary key present in the query."""
    norm = normalize(query)
    extra: list[str] = []
    for key, values in GLOSSARY.items():
        if re.search(rf"\b{re.escape(key)}\b", norm):
            extra.extend(values)
    return f"{query} {' '.join(extra)}" if extra else query
