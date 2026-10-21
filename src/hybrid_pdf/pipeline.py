"""Pipeline runner: configuration, run context and the stage protocol."""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from hybrid_pdf.errors import ExtractionError
from hybrid_pdf.model import Document, DocumentMeta

log = logging.getLogger("hybrid_pdf")

DEFAULT_STAGES = ("ingest", "naive_blocks")


@dataclass
class Config:
    llm: bool = True
    stages: list[str] = field(default_factory=lambda: list(DEFAULT_STAGES))

    @classmethod
    def from_toml(cls, path: str | Path) -> Config:
        with open(path, "rb") as f:
            data = tomllib.load(f)

        unknown = set(data) - {"llm", "stages"}

        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        
        if "llm" in data and not isinstance(data["llm"], bool):
            raise ValueError("config key 'llm' must be true or false")
        
        stages = data.get("stages", list(DEFAULT_STAGES))

        if not isinstance(stages, list) or not all(isinstance(s, str) for s in stages):
            raise ValueError("config key 'stages' must be a list of stage names")
        
        return cls(llm=data.get("llm", True), stages=stages)


@dataclass
class Context:
    """Shared state for one pipeline run. LLM client and cache are added in a later step."""

    config: Config


class Stage(Protocol):
    name: str

    def run(self, doc: Document, ctx: Context) -> Document: ...


class Pipeline:
    def __init__(self, stages: list[Stage], config: Config | None = None) -> None:
        self.stages = stages
        self.config = config or Config()

    @classmethod
    def from_config(cls, config: Config) -> Pipeline:
        from hybrid_pdf.stages import build_stages

        return cls(build_stages(config.stages), config)

    def run(self, path: str | Path) -> Document:
        doc = Document(meta=DocumentMeta(source=str(path)))
        ctx = Context(config=self.config)
        
        for stage in self.stages:
            name = getattr(stage, "name", type(stage).__name__)
            try:
                result = stage.run(doc, ctx)
            except ExtractionError:
                raise
            except Exception as exc:
                # Last-resort guard. Stages catch per-page and per-block errors themselves,
                # so reaching this means the whole stage was skipped.
                message = f"stage {name!r} failed: {exc!r}"
                log.warning(message)
                doc.meta.warnings.append(message)
            else:
                # A stage that forgets to return is a bug, not a document problem, so this
                # sits outside the try block and is never turned into a warning.
                if result is None:
                    raise TypeError(f"stage {name!r} returned None instead of a Document")
                doc = result
        return doc
