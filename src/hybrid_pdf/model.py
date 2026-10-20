"""Document model shared by all pipeline stages and exporters.

Coordinates are PDF points with the origin at the top-left corner of the page,
as PyMuPDF reports them. Page numbers are 1-based.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

BBox = tuple[float, float, float, float]  # x0, y0, x1, y1


class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    FORMULA = "formula"
    CODE = "code"
    BOILERPLATE = "boilerplate"


@dataclass
class Span:
    """A run of text with uniform font, as extracted from the PDF."""

    text: str
    bbox: BBox
    font: str
    size: float
    bold: bool
    italic: bool
    color: int  # sRGB packed as 0xRRGGBB
    page: int
    block: int  # PyMuPDF text block number within the page
    line: int  # line index within that block


@dataclass
class ImageInfo:
    bbox: BBox
    xref: int  # 0 for inline images
    width: int  # pixels
    height: int


@dataclass
class Page:
    number: int
    width: float
    height: float
    spans: list[Span] = field(default_factory=list)
    images: list[ImageInfo] = field(default_factory=list)
    drawings: list[BBox] = field(default_factory=list)  # bounding boxes of vector paths

    @property
    def has_text(self) -> bool:
        return any(span.text.strip() for span in self.spans)


@dataclass
class Provenance:
    stage: str
    method: str
    used_llm: bool = False


@dataclass
class LLMRequest:
    """A decision deferred to the LLM resolution stage."""

    kind: str
    payload: dict[str, Any]
    schema: dict[str, Any]


@dataclass
class Block:
    id: str
    type: BlockType
    text: str
    pages: list[int]
    bboxes: list[BBox]
    level: int | None = None  # headings only
    confidence: float = 1.0
    provenance: list[Provenance] = field(default_factory=list)
    section_path: list[str] = field(default_factory=list)
    llm_requests: list[LLMRequest] = field(default_factory=list)
    degraded: bool = False


@dataclass
class DocumentMeta:
    source: str
    sha256: str = ""
    page_count: int = 0
    pdf_metadata: dict[str, str] = field(default_factory=dict)
    toc: list[tuple[int, str, int]] = field(default_factory=list)  # (level, title, page)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Document:
    meta: DocumentMeta
    pages: list[Page] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)  # reading order; the source of truth

    def to_markdown(self) -> str:
        from hybrid_pdf.export.markdown import to_markdown

        return to_markdown(self)

    def to_json(self, indent: int | None = 2) -> str:
        from hybrid_pdf.export.json import to_json

        return to_json(self, indent=indent)
