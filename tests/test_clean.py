"""Stage 2 (clean.py) on small hand-made tables: no PDFs needed."""

import pandas as pd

from dnd2json import clean


def block(text, x0=50, y0=0, x1=280, y1=10, size=9.0, page=1, source="text", **extra):
    row = dict(doc_id="test.pdf", page=page, block_idx=0, text=text, x0=x0, y0=y0, x1=x1,
               y1=y1, font_size=size, is_bold=0, is_italic=0, layout_label="",
               ocr_score=float("nan"), source=source)
    row.update(extra)
    return row


def test_unwrap_joins_lines_and_keeps_bullets():
    unwrapped = clean.unwrap("You gain\nproficiency in some-\nthing.")
    assert unwrapped == "You gain proficiency in something."
    assert clean.unwrap("benefits:\n• One\n• Two") == "benefits:\n• One\n• Two"


def test_normalize_text_adds_missing_spaces_and_bullets():
    text = clean.normalize_text("tavern crowds.Where other\nbards; disadvantage.• You gain")
    assert text == "tavern crowds. Where other bards; disadvantage.\n• You gain"


def test_order_columns_reads_left_column_first():
    # Right column stored first in the file, as in "Schools of Reawakening & Hemomancy".
    df = pd.DataFrame([
        block("right top", x0=320, x1=560, y0=50, y1=60),
        block("right bottom", x0=320, x1=560, y0=300, y1=310),
        block("left top", x0=50, x1=290, y0=40, y1=50),
        block("left bottom", x0=50, x1=290, y0=200, y1=210),
    ])
    assert list(clean.order_columns(df)["text"]) == [
        "left top", "left bottom", "right top", "right bottom"]


def test_order_columns_keeps_caption_with_full_width_table():
    df = pd.DataFrame([
        block("left para", x0=50, x1=290, y0=40, y1=300),
        block("right para", x0=320, x1=560, y0=40, y1=330),
        block("Table Title", x0=50, x1=180, y0=340, y1=352),       # narrow, above the table
        block("full width table row", x0=50, x1=560, y0=360, y1=370),
    ])
    assert list(clean.order_columns(df)["text"]) == [
        "left para", "right para", "Table Title", "full width table row"]


def test_drop_cap_is_merged():
    df = pd.DataFrame([block("C", size=90), block("haotic power churns", size=9)])
    df["raw_text"] = df["text"]
    merged = clean.merge_drop_caps(df, body_size=9.0)
    assert list(merged["text"]) == ["Chaotic power churns"]


def test_split_level_lines():
    df = pd.DataFrame([block("3rd-level College of Liturgies feature You learn a cantrip.")])
    df["raw_text"] = df["text"]
    out = clean.split_level_lines(df)
    assert list(out["text"]) == ["3rd-level College of Liturgies feature", "You learn a cantrip."]


def test_clean_blocks_drops_junk_and_renumbers():
    df = pd.DataFrame([
        block("ee", source="ocr", ocr_score=0.9),                  # artwork fragment
        block("Bards who join the College of Fools use crude stories.", source="ocr",
              ocr_score=0.99),
        block("5", source="ocr", ocr_score=0.99),                  # page number
    ])
    cleaned, stats = clean.clean_blocks(df)
    assert list(cleaned["text"]) == ["Bards who join the College of Fools use crude stories."]
    assert list(cleaned["block_idx"]) == [0]
    assert stats["junk_dropped"] == 2
