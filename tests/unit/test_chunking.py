from app.rag.chunking import Document, chunk_document


def test_splits_by_heading_and_prefixes_section():
    doc = Document(doc_id="d", title="T", text="# A\n\nalpha text here\n\n## B\n\nbeta text here")
    chunks = chunk_document(doc, size=200, overlap=20)
    assert [c.section for c in chunks] == ["A", "A > B"]
    assert chunks[1].text.startswith("A > B\n")


def test_tables_are_atomic():
    table = "\n".join(["| a | b |", "|---|---|"] + [f"| row {i} with some text | value {i} |" for i in range(30)])
    doc = Document(doc_id="d", title="T", text=f"# Cmp\n\nintro paragraph\n\n{table}\n\noutro")
    chunks = chunk_document(doc, size=600, overlap=50)
    table_chunks = [c for c in chunks if "| row 0" in c.text]
    assert len(table_chunks) == 1 and "| row 29" in table_chunks[0].text


def test_overlap_never_cuts_words():
    words = " ".join(f"word{i}" for i in range(400))
    doc = Document(doc_id="d", title="T", text="\n\n".join([words[i : i + 300] for i in range(0, len(words), 300)]))
    for c in chunk_document(doc, size=500, overlap=80)[1:]:
        first = c.text.split("\n", 1)[1].split()[0]
        assert first.startswith("word") or first[0].isdigit() is False


def test_chunk_ids_are_stable():
    doc = Document(doc_id="d", title="T", text="# A\n\nsame text")
    assert chunk_document(doc)[0].chunk_id == chunk_document(doc)[0].chunk_id
