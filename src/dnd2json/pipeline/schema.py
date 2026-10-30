"""Stage 5: the shape of the output JSON.

Pydantic models do two jobs: they document the format (read this file to know what the
JSON contains), and they validate it (a missing field or wrong type raises instead of
writing a broken file).

The feature fields mirror the 5e SRD API (data/srd/5e-SRD-Subclasses.json: name, level,
description), so homebrew and official subclasses can be loaded by the same code later.
"""

from pydantic import BaseModel, ConfigDict, Field


class Feature(BaseModel):
    """One subclass feature, e.g. "Chaotic Rift" gained at level 3."""

    name: str
    level: int | None = Field(None, description="Class level that grants it; None if not found")
    description: str = Field(description="The feature's full rules text")


class ExtraSection(BaseModel):
    """Text that belongs to the subclass but isn't a feature: lore, sidebars, stat blocks."""

    name: str
    kind: str = Field(description="'lore', 'stat_block', 'spell' or 'preamble'")
    text: str


class Table(BaseModel):
    """A table from the document, as plain text rows (structure isn't recovered yet)."""

    title: str | None = None
    text: str


class Subclass(BaseModel):
    # "class" is a Python keyword, so the field is class_name and serialised as "class".
    model_config = ConfigDict(populate_by_name=True)

    name: str
    class_name: str | None = Field(None, alias="class")
    description: str = ""
    features: list[Feature] = []
    extra_sections: list[ExtraSection] = []
    tables: list[Table] = []


class Document(BaseModel):
    """Everything converted from one PDF. Most PDFs hold one subclass, some several."""

    source_file: str
    extraction: str = Field(description="'text' (PDF text layer) or 'ocr'")
    subclasses: list[Subclass]
    warnings: list[str] = Field([], description="Soft problems worth a human look")

    def to_json(self) -> str:
        # by_alias: write "class", not "class_name".
        return self.model_dump_json(by_alias=True, indent=2)
