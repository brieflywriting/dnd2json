"""Exceptions the pipeline raises on purpose.

Two kinds of problems exist:
  - Hard errors (these classes): a document can't be converted at all. The batch runner
    catches them per PDF, records them, and moves on to the next PDF.
  - Soft problems (warnings): the JSON can still be written but something looks off
    (no features found, a feature without a level...). Those are collected as strings in
    the output's `warnings` list instead of being raised.

Catching DnD2JsonError catches every pipeline error; anything else that escapes is a bug.
"""


class DnD2JsonError(Exception):
    """Base class for every error the pipeline raises on purpose."""


class ExtractionError(DnD2JsonError):
    """Stage 1 failed: the PDF couldn't be read or turned into blocks."""


class OCRUnavailableError(ExtractionError):
    """A PDF needs OCR, but the OCR environment (.venv-ocr) isn't set up."""


class NoBlocksError(DnD2JsonError):
    """A document produced no usable text blocks (empty, or everything was junk)."""


class ModelNotTrainedError(DnD2JsonError):
    """The block classifier hasn't been trained yet: run `dnd2json train`."""


class StructureError(DnD2JsonError):
    """Stage 4 couldn't build any subclass from the blocks."""
