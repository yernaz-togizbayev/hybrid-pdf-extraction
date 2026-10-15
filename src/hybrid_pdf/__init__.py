"""Hybrid PDF-to-text extraction for RAG with local LLMs."""

import logging
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hybrid-pdf-extraction")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

# Library convention: the application decides where log output goes.
logging.getLogger("hybrid_pdf").addHandler(logging.NullHandler())
