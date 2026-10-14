# Hybrid PDF Extraction: Architecture Design

**Date:** 2026-10-10
**Status:** Approved (2026-10-10)
**Related:** `DECISIONS.md` D-001 to D-003

## 1. Purpose and success criteria

The goal is a local, open-source PDF-to-text pipeline for RAG with local LLMs. It combines classical algorithms with local LLMs ("Hybrid AI") and keeps the document structure intact: sections, headings, tables and figures. It is meant for organizations that cannot send private documents to cloud AI services.

The project succeeds when:
- All five required components exist: section/paragraph detection, table extraction, image handling (OCR and description), boilerplate removal and text cleanup.
- The pipeline beats the baselines from section 6.3 on the held-out test split, on both extraction metrics and RAG metrics.
- Results in the target category (CS) are near-perfect. Weaker results in Medical and Physics are reported and discussed openly.
- Every reported number can be reproduced from the repo.
- The package can be installed from PyPI.

Out of scope for v1:
- Fully scanned PDFs, i.e. documents without a text layer that would need OCR for every page. The pipeline detects and flags such pages, and supporting them is future work.
- Documents in languages other than English.
- Cloud LLM APIs.

## 2. Constraints

- Python &ge; 3.11, with uv for environments and builds, pytest for tests and ruff for linting.
- Only open-source libraries that need no API keys, such as PyMuPDF, pdfplumber, Camelot and tabula-py.
- Local LLMs run through Ollama (`ollama-python`). The development and experiment machine has 8 GB of VRAM and 32 GB of RAM, so only one model is loaded at a time. 4B models fit comfortably, 7B to 9B models only with 4-bit quantization.
- The GitHub repo is public, so the corpus can only contain openly licensed papers (arXiv, PubMed Central open access).

## 3. Architecture overview

Classical algorithms run first, and the LLM is only called when their confidence is low (D-003). Heuristics make every decision and attach a confidence score to it. Low-confidence decisions, and tasks heuristics cannot do at all (figure descriptions, table repair), are turned into deferred LLM requests. These are processed in batches, grouped by model. Tables also get a small ensemble of extractors.

```
PDF -> 1 Ingest -> 2 Boilerplate -> 3 Regions (tables, figures) -> 4 Layout
    -> 5 Structure -> 6 Cleanup -> 7 LLM resolution -> 8 Finalize -> Export
                                                                     |- Markdown
                                                                     |- JSON
                                                                     '- RAG chunks
```

The project has three subsystems:
1. `hybrid_pdf`, the pipeline itself, published on PyPI.
2. `bench/rag`, the RAG/QA prototype.
3. `bench/`, the evaluation harness with corpus, ground truth, extractor adapters, metrics and runs.

The components diagram `diagrams/01-components.puml` shows these subsystems with their external tools.

## 4. Document model

One data structure is shared by everything. Every stage reads and writes it, and every export is generated from it. The class diagram is `diagrams/02-class-document-model.puml`.

```
Document
├── meta: DocumentMeta     source, sha256, page_count, pdf_metadata, toc, warnings
├── pages: [Page]          number, width, height, spans, images, drawings
├── blocks: [Block]        flat list in reading order
├── to_markdown(), to_json(), chunks(max_tokens) -> [Chunk]
└── sections() -> [Section]   derived tree, built from heading levels on demand

Span:       text, bbox, font, size, bold, italic, color, page, block, line
ImageInfo:  bbox, xref, width, height

Block
├── id, type, text, pages, bboxes
├── type: HEADING | PARAGRAPH | LIST_ITEM | TABLE | FIGURE | CAPTION
│         | FORMULA | CODE | BOILERPLATE
├── level            headings only
├── confidence       0..1
├── provenance       [Provenance(stage, method, used_llm)]
├── section_path     e.g. ["3", "3.2"]
├── llm_requests     [LLMRequest]
├── degraded         true if a stage failed on this block
├── table            TableData, TABLE blocks only
└── figure           FigureData, FIGURE blocks only

TableData:       cells [Cell], n_rows, n_cols, caption,
                 candidates [TableCandidate], chosen (index or None)
Cell:            row, col, text, row_span, col_span, is_header
TableCandidate:  extractor, cells [Cell], score
FigureData:      image_ref, caption, ocr_text, description

LLMRequest:  kind, payload, schema, status, result
  kind:      HEADING_DECISION | TABLE_TIEBREAK | TABLE_REPAIR
             | FIGURE_DESCRIPTION | CHART_READING
  status:    PENDING | RESOLVED | SKIPPED | FALLBACK

Section:  heading, blocks, children
Chunk:    text, section_path, pages, block_ids, block_types
```

`Span.block` and `Span.line` keep PyMuPDF's own grouping of spans into blocks and lines, which the Layout stage uses as one signal among others. The table and figure parts are added together with the Regions stage. Step 1 of the development order (section 10) builds everything else, with `LLMRequest` still lacking `status` and `result`.

Three rules apply to the model:
- The flat block list is the source of truth. The section tree is derived from the heading levels whenever it is needed.
- Boilerplate is flagged and never deleted. The exporters drop it by default, and because it is still there, the quality of boilerplate removal can be measured.
- Every change appends a provenance entry. That way each decision can be explained, and ablations can tell which stage did what.

## 5. Pipeline

### 5.1 Stage interface

```python
class Stage(Protocol):
    name: str
    def run(self, doc: Document, ctx: Context) -> Document: ...
```

The class diagram for this section is `diagrams/03-class-pipeline.puml`.

`Pipeline(stages, config)` creates an empty `Document` for the input path, builds a fresh `Context` and runs the stages in order. `Pipeline.from_config(config)` looks the stage names up in a registry.

`Config` holds the settings of a run and can be loaded from a TOML file:

```
Config:   llm (bool), stages [str], text_model, vision_model,
          thresholds {name: float}, cache_dir
Context:  config, llm (LLMClient or None), cache (ResponseCache or None),
          prompts (PromptRegistry), trace (logger)
```

The list of stages comes from the config, so an ablation run can switch off a stage or the LLM without code changes. `Context.llm` is `None` in no-LLM mode, and also when Ollama cannot be reached. The pipeline checks that once at the start of a run and warns once.

### 5.2 Stages

| # | Stage | Responsibility | Signals and tools |
|---|---|---|---|
| 1 | Ingest | Extract spans, images, vector drawings and the embedded TOC; detect pages without a text layer | PyMuPDF |
| 2 | Boilerplate | Flag running headers and footers, page numbers and watermarks | Text that repeats at the same position across pages, page-number patterns, the vertical arXiv stamp in the margin |
| 3 | Regions | Detect table and figure regions and take their spans out of the text flow. Build table candidates with the ensemble and pick one by heuristic score; a tie becomes an LLMRequest. Run OCR on text inside figures | Tables: PyMuPDF `find_tables`, pdfplumber, Camelot lattice and stream, "Table N" captions. Figures: image boxes, clusters of vector drawings, "Figure N" captions. The OCR engine is chosen by comparison and logged in DECISIONS.md |
| 4 | Layout | Detect columns and reading order, group spans into lines and lines into paragraphs, join paragraphs that continue across columns and pages | Geometry |
| 5 | Structure | Detect headings and their levels, captions, lists, formulas and code. The signals add up to a confidence score, and a low score becomes an LLMRequest | Font size compared with the most common body size, bold, numbering patterns, matches with the TOC, math fonts (CMMI, CMSY) and symbol density, monospace fonts |
| 6 | Cleanup | Undo hyphenation (checked against a word list), resolve ligatures, normalize unicode and whitespace | Deterministic rules |
| 7 | LLM resolution | Process all pending LLMRequests: the text-model batch first (heading decisions, table tie-breaks and repair), then the vision-model batch (figure descriptions, chart reading) | Ollama with JSON-schema structured output |
| 8 | Finalize | Recompute section_path and block order after the LLM corrections | |

The activity diagram `diagrams/04-activity-pipeline.puml` shows the whole run, including where low confidence turns into an LLM request.

Table extraction and OCR sit behind two small interfaces, so the tools can be swapped and compared:

```python
class TableExtractor(Protocol):
    name: str
    def extract(self, pdf: Path, page: int, region: BBox) -> list[TableCandidate]: ...

class OcrEngine(Protocol):
    def recognize(self, image: bytes) -> str: ...
```

The Regions stage holds a list of table extractors (PyMuPDF, pdfplumber, Camelot lattice, Camelot stream) and at most one OCR engine.

### 5.3 LLM handling

The LLM layer lives in the `llm` package and is used only by the LLM resolution stage:

```python
class LLMClient(Protocol):
    def generate(self, model: str, prompt: str, schema: dict,
                 images: list[bytes] = []) -> dict: ...
    def is_available(self) -> bool: ...
```

`OllamaClient` implements it through `ollama-python`, and tests use a `FakeLLMClient` with fixed answers. `ResponseCache` stores answers on disk. `PromptRegistry` returns the `Prompt` for each request kind, with its version, template and JSON schema.

- Stages never call the LLM themselves. They apply their classical decision first and then attach an `LLMRequest` to the block, with the status `PENDING`.
- Requests are grouped by model: the text-model batch runs first, then the vision-model batch, so each model is loaded once per document or batch.
- Temperature is 0 and the seed is fixed. The cache key is a hash of the model, the prompt version and the payload, so the same input gives the same answer across runs.
- Prompts are versioned files inside the package. The prompt version is part of the cache key and of the run metadata.
- A valid answer replaces the classical decision, sets the status to `RESOLVED` and adds a provenance entry with `used_llm = true`.
- If an answer is invalid or times out, the request is retried once. If it fails again, the status becomes `FALLBACK`, the classical decision stays, and a provenance note records the error.
- In no-LLM mode all requests get the status `SKIPPED` and the classical results stay. This mode is also the main ablation variant.
- The text model and the vision model are chosen by experiment on the dev split, and the choice is logged in DECISIONS.md.

The state diagram `diagrams/05-state-llm-request.puml` shows the life cycle of one request.

### 5.4 Exports

- Markdown: headings, paragraphs and GFM tables. Each figure becomes a block with its caption, OCR text and description. Boilerplate is left out.
- JSON: the full document model, which the evaluation reads.
- Chunks: a `Chunker(max_tokens)` turns the document into `Chunk` objects that follow the section structure and carry metadata (section_path, pages, block ids, block types). A table or figure is never split, and no chunk crosses the boundary between two top-level sections.

### 5.5 Public API and CLI

```python
from hybrid_pdf import extract
doc = extract("paper.pdf", llm=True)          # Document
doc.to_markdown(); doc.to_json(); doc.chunks(max_tokens=512)
```

```
hybrid-pdf extract paper.pdf --format md|json|chunks [--no-llm] [--config cfg.toml]
```

## 6. Evaluation (`bench/`)

### 6.1 Corpus

- Three categories (D-001, provisional): CS as the target, plus Medical and Physics. CS and Physics papers come from arXiv, Medical papers from PMC open access.
- `data/manifest.csv` has the columns `id, category, source, source_id, version, license, pdf_url, source_url, sha256, split`.
- PDFs and source files are not committed. `bench corpus download` fetches them and verifies the hashes.
- The corpus starts with about 30 papers per category, 10 for dev and 20 for test. All tuning happens on dev, and all reported results come from test.
- Papers without usable source files are excluded, and every exclusion is documented.

### 6.2 Ground truth (D-002)

- LaTeX from arXiv is converted with pandoc or LaTeXML (the two will be compared). JATS XML from PMC is converted with lxml. Both paths produce the same ground-truth format: headings with levels, body text in order, table cell grids and figure captions.
- About 5 papers per category are checked by hand, and the error rate found in this sample is reported.

### 6.3 Systems compared

Every system implements `Extractor.extract(pdf) -> ExtractorOutput(text_or_markdown, tables?)`.
- Baselines: PyMuPDF `get_text`, pymupdf4llm, pdfplumber, Camelot and tabula (tables only), the LlamaIndex PDF reader, and LangChain's `PyPDFLoader` and `PyMuPDFLoader`.
- Our systems: the full pipeline, the no-LLM variant and ablations that remove one stage each.

### 6.4 Extraction metrics

| Aspect | Metric |
|---|---|
| Text | Normalized edit distance of the body text, reading-order accuracy |
| Structure | Heading precision, recall and F1; accuracy of heading levels |
| Tables | TEDS, cell-level F1 |
| Boilerplate | Share of boilerplate left in the output |
| Figures | Caption recall (description quality is measured through QA) |

### 6.5 RAG prototype and QA

- The prototype is a CLI with the commands `bench rag index <system>` and `bench rag ask`.
- Every system gets the same setup: the same local embedding model (sentence-transformers), the same vector store and the same Ollama model for generating answers.
- To keep the comparison fair, every system's output goes through the same generic chunker. Our section-aware chunker runs as a separate variant, so the effect of text quality can be told apart from the effect of chunking.
- Each QA pair points to a location in the ground truth. There are four question types: a fact from the text, a value from a table, a question about a figure, and a fact from a particular section. Table questions are generated from templates. Text questions are drafted by an LLM from the ground truth and then filtered by hand.
- Metrics:
  - Retrieval recall@k, i.e. whether the retrieved chunks contain the gold evidence. This metric isolates the effect of extraction.
  - Answer accuracy by exact or numeric match. A local LLM judge serves as a secondary metric, and its verdicts are spot-checked by hand.

### 6.6 Runs

Each run writes `results/<run_id>/` with the config, library versions, git commit, model names, prompt versions, the output for each document and the metrics. Every number in the thesis is produced by a script in `bench/`.

## 7. Repository layout

```
src/hybrid_pdf/          PyPI package
  model.py               Document, Page, Span, Block, LLMRequest
  pipeline.py            Pipeline, Stage, Context, config
  stages/                ingest, boilerplate, regions/, layout, structure, cleanup, llm, finalize
  llm/                   Ollama client, cache, prompts/
  export/                markdown, json, chunks
  cli.py
bench/                   not packaged
  corpus/ groundtruth/ extractors/ metrics/ rag/ qa/ runs.py
tests/
  unit/                  synthetic PDFs generated in code
  integration/           a few real CC-BY papers with golden outputs
data/manifest.csv
results/                 gitignored, except summary tables
```

Imports follow four rules. Stages never import each other. Only the LLM resolution stage uses the `llm` package; all other stages attach requests to blocks. From `hybrid_pdf`, `bench` uses only the public API, and `hybrid_pdf` never imports `bench`.

There are two optional extras, `hybrid-pdf-extraction[ocr]` and `[tables]` (Camelot and tabula; tabula needs Java). The dependencies of `bench` live in a separate dependency group and are not a package extra.

## 8. Error handling

- If a stage raises an exception on a page or block, the exception is caught and the classical or previous result is kept. The block gets `degraded=True`, the error goes into the trace and a warning goes into `doc.meta.warnings`.
- If Ollama is not reachable, the pipeline warns once and continues in no-LLM mode.
- An encrypted or corrupt PDF raises `ExtractionError` with a clear message.
- A page without a text layer gets a warning in the metadata and is skipped in v1.

## 9. Testing

- Each stage has unit tests on synthetic PDFs generated in code, with controlled fonts, columns, headers and tables.
- The LLM stage is tested with a fake client that returns fixed responses, including invalid ones.
- Integration tests use 3 to 5 real CC-BY papers with golden JSON and Markdown snapshots.
- During development the dev-split metrics act as a regression check.
- Heuristics and exporters are written test-first.

## 10. Development order

1. Package skeleton, document model, pipeline, Ingest, plain Markdown export and CLI.
2. Corpus manifest and downloader, a ground-truth builder for arXiv LaTeX, a minimal set of metrics and two baselines. After this step the first end-to-end numbers exist.
3. RAG prototype and the first QA set.
4. The components one at a time, in this order: boilerplate, layout, structure, cleanup, regions and tables, figures, LLM resolution. Each one is measured against the baselines on dev, and each decision is logged.
5. JATS ground truth, the remaining baselines, the full QA set, the final runs on the test split and the ablations.

Each step gets its own implementation plan.

## 11. Open questions

- How many categories: to be confirmed with Dr. Lux (D-001).
- Which text and vision models: chosen on dev data.
- Which OCR engine: Tesseract or a pip-installable alternative such as RapidOCR, compared while building the figures component.
- Which LaTeX converter: pandoc or LaTeXML, compared during the ground-truth work.
