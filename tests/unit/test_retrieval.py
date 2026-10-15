"""Golden retrieval set: each query must retrieve its source document in the top 3.

This is the RAG regression gate: if a corpus/chunking/embedding change drops recall,
CI fails before the LLM ever sees worse context.
"""

import pytest

GOLDEN = [
    ("What's the difference between the Get Clássica and the Get Smart?", "seed:02-card-machines"),
    ("Do I need a bank account to receive my sales via Pix?", "seed:03-pix"),
    ("When is the money from my sales deposited?", "seed:04-settlement-terms"),
    ("How does receivables advance (antecipação) work with Getnet?", "seed:05-receivables-advance"),
    ("Can I sell through WhatsApp using the Payment Link?", "seed:06-payment-link-whatsapp"),
    ("How many installments can I split a sale into with the crediário?", "seed:07-crediario"),
    ("My card machine won't connect to the internet, what should I do?", "seed:08-troubleshooting-connectivity"),
    ("My card machine is showing a transaction decline error.", "seed:09-decline-codes"),
    ("Qual a diferença entre a Get Clássica e a Get Smart?", "seed:02-card-machines"),
    ("Como funciona a antecipação de recebíveis?", "seed:05-receivables-advance"),
    ("minha maquininha não conecta na internet", "seed:08-troubleshooting-connectivity"),
    ("em quantas parcelas posso vender no crediário?", "seed:07-crediario"),
    ("what is the whatsapp number of getnet support", "seed:10-support-channels"),
    ("how much is the rent of the Get Mini", "seed:02-card-machines"),
]


@pytest.mark.parametrize("query,doc_id", GOLDEN)
def test_golden_queries_hit_top3(kb, query, doc_id):
    hits = kb.search(query, top_k=3)
    assert doc_id in [h.chunk.doc_id for h in hits], [h.chunk.doc_id for h in hits]


def test_recall_at_1_is_high(kb):
    top1 = sum(kb.search(q, top_k=1)[0].chunk.doc_id == d for q, d in GOLDEN)
    assert top1 / len(GOLDEN) >= 0.8


def test_results_are_diversified(kb):
    hits = kb.search("Get Smart Get Clássica Get Mini", top_k=5, per_doc=2)
    assert max(sum(h.chunk.doc_id == d for h in hits) for d in {h.chunk.doc_id for h in hits}) <= 2


def test_store_roundtrip(kb, tmp_path):
    from app.rag.store import VectorStore

    kb.save(tmp_path)
    again = VectorStore.load(tmp_path)
    assert len(again.chunks) == len(kb.chunks)
    assert again.search("Pix")[0].chunk.chunk_id == kb.search("Pix")[0].chunk.chunk_id
