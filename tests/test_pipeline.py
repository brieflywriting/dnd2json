"""The whole pipeline on a real PDF, plus the output format and error handling.

Skipped automatically when the PDF or the trained model isn't there (fresh checkout).
"""

import json

import pytest

from dnd2json import config
from dnd2json.errors import ExtractionError
from dnd2json.pipeline.convert import convert_many, convert_pdf, ensure_blocks
from dnd2json.pipeline.schema import Document, Feature, Subclass

SPELL_THIEF = config.PDF_DIR / "Wizard - Spell Thief.pdf"  # text-layer PDF: no OCR needed

needs_pdf = pytest.mark.skipif(not SPELL_THIEF.exists(), reason="PDF not available")
needs_model = pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="run `dnd2json train`")


def test_json_uses_class_not_class_name():
    doc = Document(source_file="x.pdf", extraction="text", subclasses=[
        Subclass(name="X", features=[Feature(name="F", level=3, description="d")],
                 **{"class": "Bard"})])
    data = json.loads(doc.to_json())
    assert data["subclasses"][0]["class"] == "Bard"
    assert "class_name" not in data["subclasses"][0]


def test_unreadable_pdf_raises_extraction_error(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is not a pdf")
    with pytest.raises(ExtractionError):
        ensure_blocks(broken, blocks_dir=tmp_path)


@needs_pdf
@needs_model
def test_spell_thief_end_to_end():
    from dnd2json.classifier.block_classifier import BlockClassifier
    doc = convert_pdf(SPELL_THIEF, BlockClassifier())
    (sub,) = doc.subclasses
    assert sub.name == "Spell Thief" and sub.class_name == "Wizard"
    assert sub.description.startswith("These wizards specialise")
    assert [(f.name, f.level) for f in sub.features] == [
        ("Arcane Vault", 2), ("Stolen Power", 2), ("Arcana Laundering", 6),
        ("Loot Magic", 10), ("Arcana Abduction", 14)]


@needs_model
def test_batch_keeps_going_after_a_bad_pdf(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is not a pdf")
    pdfs = [broken] + ([SPELL_THIEF] if SPELL_THIEF.exists() else [])
    report = convert_many(pdfs, out_dir=tmp_path / "out")
    assert report[0]["status"] == "failed" and "ExtractionError" in report[0]["error"]
    if SPELL_THIEF.exists():
        assert report[1]["status"] == "ok"
    assert (tmp_path / "out" / "_report.json").exists()
