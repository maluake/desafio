"""Tiny, dependency-free language identification (PT vs EN) for routing and offline replies."""

from __future__ import annotations

from app.rag.text import normalize

_PT = set(
    """
    o a os as um uma de do da dos das em no na que nao sim para por com meu minha minhas meus
    voce como quando qual quais onde porque maquininha vendas dinheiro ontem hoje amanha conta
    esta estao preciso quero posso tem ter fazer cai caiu receber recebo parcelas taxa obrigado
    ola oi
    """.split()
)
_EN = set(
    """
    the a an of in on that not yes for by with my mine you how when what which where why machine
    sales money yesterday today tomorrow account is are need want can have do will receive
    installments fee thanks hello hi
    """.split()
)


def detect_language(text: str) -> str:
    words = normalize(text).replace("?", " ").replace(",", " ").split()
    pt = sum(w in _PT for w in words)
    en = sum(w in _EN for w in words)
    if any(ch in text for ch in "ãõçáéíóúâêô"):
        pt += 1
    return "pt" if pt > en else "en"


def t(lang: str, en: str, pt: str) -> str:
    return pt if lang == "pt" else en
