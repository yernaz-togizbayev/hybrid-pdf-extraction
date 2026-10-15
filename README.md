# hybrid-pdf-extraction

PDF-to-text extraction for Retrieval-Augmented Generation (RAG) with local LLMs. Classical layout analysis does most of the work, and local models run through Ollama help where the heuristics are unsure. The output keeps the document structure: sections, headings, tables and figures.

This is a bachelor thesis project at the University of Vienna and is in early development.

## Development

```bash
uv sync                    # create .venv and install the package with dev tools
uv run pytest              # tests
uv run ruff check .        # lint
uv run ruff format .       # format
```

## License

AGPL-3.0-or-later, because the package depends on PyMuPDF, which is AGPL licensed. See `LICENSE` and decision D-004 in `DECISIONS.md`.
