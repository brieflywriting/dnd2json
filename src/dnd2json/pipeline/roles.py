"""Stage 3a: give every cleaned block a layout role.

Roles (what the block IS on the page, not what it means for the subclass):
  heading     a short line set apart by size/boldness (feature names, titles)
  level_line  "3rd-level College of Fools feature": announces the level of the feature above
  body        paragraph text
  table       table rows, stat block lines, feature-summary tables
  other       page furniture: credits, legal text, page headers/footers, group headers

Rules are tried in order and the first match wins. Each block also gets `role_reason`,
a short note on which rule fired. When a block ends up in the wrong place in the JSON,
look at its role and reason in the debug CSV (`dnd2json convert --debug`) first.

Whether a body block is a description or a feature is NOT decided here: that's the
classifier's job (classifier.py), using the words, not the layout.
"""

import pandas as pd

from dnd2json import patterns

# OCR layout labels that are never subclass content.
OCR_OTHER_LABELS = {
    "header", "footer", "number", "footnote", "vision_footnote", "header_image",
    "footer_image", "image", "seal", "formula", "aside_text_image",
}
OCR_HEADING_LABELS = {"paragraph_title", "doc_title"}
# Artwork read as text is a few garbled words; a real text box is paragraphs long.
OCR_IMAGE_TEXT_MIN_CHARS = 150
OCR_TABLE_LABELS = {"table", "figure_title", "chart", "chart_title", "table_title"}

# Text-layer headings: noticeably bigger than body text, or bold at body size.
HEADING_SIZE_RATIO = 1.15
# Headings are short: a feature name, not a sentence.
HEADING_MAX_WORDS = 12
HEADING_MAX_CHARS = 90
# OCR "text" blocks this short, mostly upper-case and without a full stop are headings the
# layout model missed ("BARDIC MASTERPIECE").
OCR_HEADING_MAX_WORDS = 6
OCR_HEADING_UPPER_RATIO = 0.6


def _is_short_line(text: str) -> bool:
    words = text.split()
    return (0 < len(words) <= HEADING_MAX_WORDS and len(text) <= HEADING_MAX_CHARS
            and "\n" not in text and not text.rstrip().endswith((".", ",", ";")))


def _upper_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


def classify_role(row, body_size: float) -> tuple[str, str]:
    """(role, reason) for one block. The order of the checks is the priority."""
    text = row["text"]
    label = row.get("layout_label", "") or ""
    size = row["font_size"]

    # --- level announcements ---------------------------------------------------------------
    # First: the wording is unambiguous, and the OCR layout model sometimes files level
    # lines under "vision_footnote" or "header", which would throw them away.
    if patterns.LEVEL_LINE.match(text):
        return "level_line", "level line pattern"

    # --- page furniture ------------------------------------------------------------------
    # Long text inside an "image" region is a decorated text box (a feature set on a
    # parchment background), not artwork: keep it as body text.
    if label == "image" and len(text) >= OCR_IMAGE_TEXT_MIN_CHARS:
        return "body", "long text in ocr image region"
    if label in OCR_OTHER_LABELS:
        return "other", f"ocr label {label}"
    if len(text) <= patterns.CREDITS_MAX_CHARS and patterns.CREDITS.search(text):
        return "other", "credits/legal text"
    if patterns.SECTION_HEADER.match(text.strip()):
        return "other", "group header"

    # --- tables ------------------------------------------------------------------------------
    if label in OCR_TABLE_LABELS:
        return "table", f"ocr label {label}"

    # --- headings ---------------------------------------------------------------------------
    if row["source"] == "ocr":
        if label in OCR_HEADING_LABELS and _is_short_line(text):
            return "heading", f"ocr label {label}"
        if (_is_short_line(text) and len(text.split()) <= OCR_HEADING_MAX_WORDS
                and _upper_ratio(text) >= OCR_HEADING_UPPER_RATIO):
            return "heading", "short upper-case line"
    elif pd.notna(size) and _is_short_line(text):
        if size >= body_size * HEADING_SIZE_RATIO:
            return "heading", f"size {size:g} > body {body_size:g}"
        # Bold at body size is a heading; bold at another size is a table's header row
        # ("Armor Class 13", "STR DEX CON INT WIS CHA" in a stat block).
        if row["is_bold"] and round(size) == round(body_size):
            return "heading", "bold short line at body size"

    # --- text-layer tables: not body size, not a heading -------------------------------------
    # Tables and stat blocks use their own (usually smaller or slightly bigger) font.
    if row["source"] == "text" and pd.notna(size) and round(size) != round(body_size):
        return "table", f"size {size:g} != body {body_size:g}"

    return "body", "default"


def assign_roles(df: pd.DataFrame, body_size: float) -> pd.DataFrame:
    """Add `role` and `role_reason` columns to the cleaned blocks."""
    df = df.copy()
    results = [classify_role(row, body_size) for _, row in df.iterrows()]
    df["role"] = [role for role, _ in results]
    df["role_reason"] = [reason for _, reason in results]
    df = _mark_repeated_furniture(df)
    return df


# Text that repeats on several pages (running headers like "CHAPTER 2 | SUBCLASSES",
# an author's name in every footer) is page furniture, whatever its other features.
REPEATED_MIN_PAGES = 2


def _mark_repeated_furniture(df: pd.DataFrame) -> pd.DataFrame:
    if df["page"].nunique() < REPEATED_MIN_PAGES:
        return df
    short = df["text"].str.len() <= HEADING_MAX_CHARS
    pages_per_text = df[short].groupby("text")["page"].nunique()
    repeated = set(pages_per_text[pages_per_text >= REPEATED_MIN_PAGES].index)
    # Some short text legitimately repeats: sub-headings ("Mask of Comedy" under two
    # features) and level lines ("3rd-level College of Drama feature" for two features).
    hit = df["text"].isin(repeated) & ~df["role"].isin(["heading", "level_line"])
    df.loc[hit, ["role", "role_reason"]] = ["other", "repeated on several pages"]
    return df
