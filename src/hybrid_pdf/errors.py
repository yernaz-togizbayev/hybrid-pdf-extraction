"""Exceptions raised by the public API."""


class ExtractionError(Exception):
    """The PDF cannot be processed at all (missing, corrupt or encrypted)."""
