import pytest

from hybrid_pdf.errors import ExtractionError
from hybrid_pdf.model import Document
from hybrid_pdf.pipeline import DEFAULT_STAGES, Config, Context, Pipeline


class RecordingStage:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    def run(self, doc: Document, ctx: Context) -> Document:
        self.calls.append(self.name)
        return doc


class FailingStage:
    name = "boom"

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def run(self, doc: Document, ctx: Context) -> Document:
        raise self.exc


class NamelessFailingStage:
    def run(self, doc: Document, ctx: Context) -> Document:
        raise RuntimeError("bad page")


class NoneReturningStage:
    name = "forgetful"

    def run(self, doc: Document, ctx: Context) -> Document:
        return None  # type: ignore[return-value]


def test_default_config():
    config = Config()
    assert config.llm is True
    assert config.stages == list(DEFAULT_STAGES)


def test_config_from_toml(tmp_path):
    path = tmp_path / "cfg.toml"
    path.write_text('llm = false\nstages = ["ingest"]\n', encoding="utf-8")
    assert Config.from_toml(path) == Config(llm=False, stages=["ingest"])


def test_config_from_toml_rejects_unknown_key(tmp_path):
    path = tmp_path / "cfg.toml"
    path.write_text("lmm = false\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lmm"):
        Config.from_toml(path)


@pytest.mark.parametrize("content", ['llm = "yes"\n', 'stages = "ingest"\n', "stages = [1]\n"])
def test_config_from_toml_rejects_wrong_types(tmp_path, content):
    path = tmp_path / "cfg.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        Config.from_toml(path)


def test_pipeline_runs_stages_in_order_and_sets_source():
    calls: list[str] = []
    pipeline = Pipeline([RecordingStage("a", calls), RecordingStage("b", calls)])
    doc = pipeline.run("paper.pdf")
    assert calls == ["a", "b"]
    assert doc.meta.source == "paper.pdf"


def test_failing_stage_becomes_warning_and_later_stages_still_run():
    calls: list[str] = []
    pipeline = Pipeline([FailingStage(RuntimeError("bad page")), RecordingStage("after", calls)])
    doc = pipeline.run("paper.pdf")
    assert calls == ["after"]
    assert len(doc.meta.warnings) == 1
    assert "boom" in doc.meta.warnings[0] and "bad page" in doc.meta.warnings[0]


def test_extraction_error_propagates():
    pipeline = Pipeline([FailingStage(ExtractionError("encrypted"))])
    with pytest.raises(ExtractionError, match="encrypted"):
        pipeline.run("paper.pdf")


def test_failing_stage_without_name_still_becomes_warning():
    pipeline = Pipeline([NamelessFailingStage()])
    doc = pipeline.run("paper.pdf")
    assert len(doc.meta.warnings) == 1
    assert "NamelessFailingStage" in doc.meta.warnings[0] and "bad page" in doc.meta.warnings[0]


def test_stage_returning_none_is_a_type_error():
    pipeline = Pipeline([NoneReturningStage()])
    with pytest.raises(TypeError, match="'forgetful' returned None"):
        pipeline.run("paper.pdf")
