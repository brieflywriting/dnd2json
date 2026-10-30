"""Stage 2: make extracted blocks usable.

Input: the blocks CSV from stage 1 (extract.py). Output: the same table, but
  1. in real reading order (two-column pages read column by column),
  2. without junk (OCR fragments of artwork, empty blocks),
  3. with drop caps glued back onto their word ("C" + "haotic" -> "Chaotic"),
  4. with repaired text: lines unwrapped, OCR digit errors fixed, missing spaces added,
  5. with level lines split off the paragraph they were glued to.

The original text is kept in `raw_text`, because tables need their line breaks.
"""

import math
import re

import pandas as pd

from dnd2json.errors import NoBlocksError
from dnd2json.pipeline import patterns

# --- Reading order ------------------------------------------------------------------------
# A block belongs to the left column if it ends before the page middle (plus this slack),
# to the right column if it starts after the middle (minus slack), otherwise it spans both.
COLUMN_SLACK_PT = 15

# --- Junk ---------------------------------------------------------------------------------
MIN_ALNUM_CHARS = 4         # fewer letters/digits than this: a fragment, not content
OCR_JUNK_SCORE = 0.6        # OCR blocks below this confidence...
OCR_JUNK_MAX_CHARS = 40     # ...and shorter than this are artwork read as text

# --- Drop caps ----------------------------------------------------------------------------
DROP_CAP_MAX_CHARS = 2      # "C", "D" (sometimes with a stray mark)
DROP_CAP_SIZE_RATIO = 1.8   # a drop cap is far bigger than body text (90 pt vs 9 pt)

# Blocks this long count as paragraphs when finding the body text size.
PARAGRAPH_MIN_CHARS = 80


def load_blocks(csv_path) -> pd.DataFrame:
    """Read a stage-1 CSV. Empty cells stay empty strings, not NaN, for easy string ops."""
    df = pd.read_csv(csv_path, keep_default_na=False, encoding="utf-8")
    if df.empty:
        raise NoBlocksError(f"{csv_path} has no blocks")
    # Numeric columns that may contain "" (OCR rows have no bold flag, etc.).
    for column in ["x0", "y0", "x1", "y1", "font_size", "ocr_score"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ["is_bold", "is_italic"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
    df["text"] = df["text"].astype(str)
    return df


def body_font_size(df: pd.DataFrame) -> float:
    """The size of normal paragraph text in this document.

    Taken as the size covering the most characters among paragraph-length blocks:
    headings and tables are rarer and shorter, so they can't win.
    """
    paragraphs = df[df["text"].str.len() >= PARAGRAPH_MIN_CHARS].dropna(subset=["font_size"])
    if paragraphs.empty:
        paragraphs = df.dropna(subset=["font_size"])
    if paragraphs.empty:
        return float("nan")
    chars_per_size = paragraphs.groupby(paragraphs["font_size"].round())["text"].apply(
        lambda texts: texts.str.len().sum())
    return float(chars_per_size.idxmax())


# =========================================================================================
# 1. Reading order
# =========================================================================================
def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder text-layer blocks page by page into column reading order.

    Many PDFs store their content in reading order already, but some store the right
    column first (e.g. "Schools of Reawakening & Hemomancy"). Rebuilding the order from
    positions fixes both:
      - walk the page top to bottom;
      - a block spanning both columns (a title, a full-width table) closes a "band";
      - inside a band, read the whole left column, then the whole right column.
    OCR blocks are left alone: the layout model already returns them in reading order.
    """
    ordered_pages = []
    for _, page in df.groupby("page", sort=True):
        middle = (page["x0"].min() + page["x1"].max()) / 2
        left = page["x1"] <= middle + COLUMN_SLACK_PT
        right = page["x0"] >= middle - COLUMN_SLACK_PT
        column = pd.Series("span", index=page.index)
        column[left] = "left"
        column[right & ~left] = "right"

        page = page.assign(_column=column).sort_values("y0", kind="stable")
        bands, current = [], []
        for idx, row in page.iterrows():
            if row["_column"] == "span":
                # A narrow title right above a full-width table ("Destructive Spell Table")
                # belongs to the table, not to its column: move it into the span band.
                captions = []
                while current and _is_caption_of(page, current, current[-1], row):
                    captions.insert(0, current.pop())
                bands.append(current)
                bands.append(captions + [idx])
                current = []
            else:
                current.append(idx)
        bands.append(current)

        order = []
        for band in bands:
            band_rows = page.loc[band]
            order += list(band_rows[band_rows["_column"] != "right"].index)  # left (and span)
            order += list(band_rows[band_rows["_column"] == "right"].index)
        ordered_pages.append(page.loc[order].drop(columns="_column"))
    return pd.concat(ordered_pages)


CAPTION_MAX_GAP_PT = 30  # a caption sits at most this far above what it captions


def _is_caption_of(page, band, idx, span_row) -> bool:
    """Is block `idx` a caption sitting directly above the full-width block `span_row`?

    True if it's close above it and lower than everything in the other column, i.e.
    nothing in the band is read after it anyway, so moving it can't break the columns.
    """
    block = page.loc[idx]
    if span_row["y0"] - block["y1"] > CAPTION_MAX_GAP_PT:
        return False
    others = page.loc[[i for i in band if page.loc[i, "_column"] != block["_column"]]]
    return others.empty or block["y0"] >= others["y1"].max()


# =========================================================================================
# 2. Junk
# =========================================================================================
def is_junk(row) -> bool:
    text = row["text"]
    if sum(c.isalnum() for c in text) < MIN_ALNUM_CHARS:  # page numbers, stray marks
        return True
    if row["source"] == "ocr":
        score = row["ocr_score"]
        if pd.notna(score) and score < OCR_JUNK_SCORE and len(text) < OCR_JUNK_MAX_CHARS:
            return True
    return False


# =========================================================================================
# 3. Drop caps
# =========================================================================================
def merge_drop_caps(df: pd.DataFrame, body_size: float) -> pd.DataFrame:
    """Glue a giant first letter onto the paragraph that follows it.

    Text-layer PDFs store a decorative drop cap as its own block: "C" (90 pt), then
    "haotic power churns..." (9 pt). If a block is one or two letters, far bigger than
    body text, and the next block starts in lowercase, they're one word.
    """
    rows = df.to_dict("records")
    merged, skip_next = [], False
    for i, row in enumerate(rows):
        if skip_next:
            skip_next = False
            continue
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        text = row["text"].strip()
        is_cap = (
            nxt is not None
            and 0 < len(text) <= DROP_CAP_MAX_CHARS
            and text[0].isalpha() and text[0].isupper()
            and pd.notna(row["font_size"]) and not math.isnan(body_size)
            and row["font_size"] >= body_size * DROP_CAP_SIZE_RATIO
            and nxt["text"][:1].islower()
        )
        if is_cap:
            nxt = dict(nxt)
            nxt["text"] = text[0] + nxt["text"]
            nxt["raw_text"] = text[0] + nxt["raw_text"]
            merged.append(nxt)
            skip_next = True
        else:
            merged.append(row)
    return pd.DataFrame(merged, columns=df.columns)


# =========================================================================================
# 4. Text repair
# =========================================================================================
def repair_ocr(text: str) -> str:
    """Fix OCR misreads that break level detection: "1Oth" -> "10th", "1 4th" -> "14th"."""
    text = patterns.OCR_O_IN_NUMBER.sub("0", text)
    text = patterns.OCR_O_BEFORE_UNIT.sub("0", text)
    text = patterns.OCR_SPLIT_ORDINAL.sub(r"1\1", text)
    return text


def unwrap(text: str) -> str:
    """Join the visual lines of a paragraph into one line of text.

    PDFs break paragraphs into lines to fit the column. Bullets stay on their own line;
    a word hyphenated across lines is rejoined ("some-\\nthing" -> "something").
    """
    text = patterns.HYPHEN_BREAK.sub("", text)
    text = patterns.INLINE_BULLET.sub("\n• ", text)
    lines = [line.strip() for line in text.split("\n")]
    out = []
    for line in lines:
        if not line:
            continue
        if out and not line.startswith("•"):
            out[-1] += " " + line
        else:
            out.append(line)
    return "\n".join(out)


def normalize_text(text: str) -> str:
    text = repair_ocr(text)
    text = unwrap(text)
    text = patterns.MISSING_SPACE.sub(" ", text)
    return re.sub(r"[ \t]+", " ", text).strip()  # collapse runs of spaces, keep newlines


def split_level_lines(df: pd.DataFrame) -> pd.DataFrame:
    """Split "3rd-level X feature You learn..." into the level line and the paragraph.

    Text-layer PDFs set the level line in the body font, so PyMuPDF puts it in the same
    block as the paragraph after it. Two rows make the level line visible to roles.py.
    """
    rows = []
    for row in df.to_dict("records"):
        match = patterns.LEADING_LEVEL_LINE.match(row["text"])
        if match and match.end() < len(row["text"]):
            rows.append({**row, "text": match.group().strip(), "raw_text": match.group().strip()})
            rest = row["text"][match.end():].strip()
            rows.append({**row, "text": rest, "raw_text": rest})
        else:
            rows.append(row)
    return pd.DataFrame(rows, columns=df.columns)


# =========================================================================================
# Stage entry point
# =========================================================================================
def clean_blocks(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run all cleaning steps. Returns the cleaned blocks and counts of what was done."""
    df = df.copy()
    source = df["source"].iloc[0]
    df["raw_text"] = df["text"].map(repair_ocr)
    df["text"] = df["text"].map(normalize_text)
    stats = {"blocks_in": len(df)}

    if source == "text":
        df = order_columns(df)

    body_size = body_font_size(df)
    before = len(df)
    df = merge_drop_caps(df.reset_index(drop=True), body_size)
    stats["drop_caps_merged"] = before - len(df)

    junk = df.apply(is_junk, axis=1)
    stats["junk_dropped"] = int(junk.sum())
    df = df[~junk].reset_index(drop=True)
    if df.empty:
        raise NoBlocksError("every block was junk")

    before = len(df)
    df = split_level_lines(df)
    stats["level_lines_split"] = len(df) - before

    df["block_idx"] = range(len(df))  # new reading order
    stats["blocks_out"] = len(df)
    stats["body_font_size"] = body_size
    return df, stats
