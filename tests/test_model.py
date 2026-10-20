from hybrid_pdf.model import Block, BlockType, Document, DocumentMeta, Page, Span


def make_span(text: str) -> Span:
    return Span(
        text=text,
        bbox=(0, 0, 10, 10),
        font="Helvetica",
        size=10.0,
        bold=False,
        italic=False,
        color=0,
        page=1,
        block=0,
        line=0,
    )


def test_page_without_spans_has_no_text():
    assert not Page(number=1, width=595, height=842).has_text


def test_page_with_only_whitespace_spans_has_no_text():
    page = Page(number=1, width=595, height=842, spans=[make_span("  "), make_span("\n")])
    assert not page.has_text


def test_page_with_visible_text_has_text():
    page = Page(number=1, width=595, height=842, spans=[make_span(" "), make_span("Hello")])
    assert page.has_text


def test_new_document_is_empty():
    doc = Document(meta=DocumentMeta(source="paper.pdf"))
    assert doc.pages == [] and doc.blocks == [] and doc.meta.warnings == []


def test_block_defaults():
    block = Block(id="b1", type=BlockType.PARAGRAPH, text="x", pages=[1], bboxes=[(0, 0, 1, 1)])
    assert block.confidence == 1.0
    assert block.level is None
    assert not block.degraded
    assert block.provenance == [] and block.llm_requests == [] and block.section_path == []


def test_block_type_values_are_plain_strings():
    assert BlockType.LIST_ITEM == "list_item"
