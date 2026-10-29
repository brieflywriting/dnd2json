"""Stage 1: PDF -> one row per text block (CSV), in reading order.

Each PDF goes down one of two paths, and both write the same columns:
  - "text": the PDF has a usable text layer -> PyMuPDF blocks with font size and style.
  - "ocr":  scanned or garbled PDF          -> PaddleOCR PP-StructureV3 layout blocks.

Run (text-layer PDFs work on any Python; OCR needs the Python 3.12 env, see README):
    python src/dnd2json/extract.py --only text
    .venv-ocr/Scripts/python src/dnd2json/extract.py --only ocr

This file deliberately imports nothing from the dnd2json package, so the OCR environment
can run it without installing the rest of the project.
"""

import argparse
import csv
import functools
import json
import sys
import traceback
from collections import Counter
from pathlib import Path

import pymupdf

# Bump when the output format or extraction logic changes: outputs made with another
# version (or other settings) are redone instead of skipped.
EXTRACTOR_VERSION = 2

COLUMNS = [
    "doc_id", "page", "block_idx", "text",
    "x0", "y0", "x1", "y1",  # bbox in PDF points (1/72 inch), origin top-left
    "font_size",             # text: dominant span size; ocr: median line height in points
    "is_bold", "is_italic",  # text path only
    "layout_label",          # ocr path only (paragraph_title, text, table, footer...)
    "ocr_score",             # ocr path only: mean recognition confidence of its lines
    "source",                # "text" or "ocr"
]


def log(message: str) -> None:
    # flush: progress shows up immediately and survives a crash, even when piped to a file.
    print(message, flush=True)


# =========================================================================================
# Routing: text layer or OCR?
# =========================================================================================
MIN_PAGE_CHARS = 10      # fewer than this: the page has no real text layer
MIN_LETTER_RATIO = 0.7    # below this, a page's text is mostly junk
MAX_GARBAGE_RATIO = 0.05  # replacement, control and undecodable private-use characters

REPLACEMENT_CHAR = "�"         # "�": a character lost in an encoding round-trip
FIRST_PRINTABLE = 0x20              # code points below this are control characters
ALLOWED_CONTROL = "\n\t\r"
PRIVATE_USE = range(0xE000, 0xF8FF + 1)  # Unicode Private Use Area: font-specific glyphs

# Symbol fonts put character X at SYMBOL_FONT_OFFSET + ord(X), e.g. "" = "B".
SYMBOL_FONT_OFFSET = 0xF000
SYMBOL_FONT_RANGE = range(SYMBOL_FONT_OFFSET + 0x20, SYMBOL_FONT_OFFSET + 0xFF + 1)


def decode_symbol_font(text: str) -> str:
    """Map symbol-font private-use characters back to ASCII.

    Some PDFs embed "symbol" fonts whose characters land in the private-use area at
    SYMBOL_FONT_OFFSET + the real code. Subtracting the offset recovers the text.
    """
    return "".join(
        chr(ord(c) - SYMBOL_FONT_OFFSET) if ord(c) in SYMBOL_FONT_RANGE else c for c in text
    )


def is_garbage(c: str) -> bool:
    return (
        c == REPLACEMENT_CHAR
        or (ord(c) < FIRST_PRINTABLE and c not in ALLOWED_CONTROL)
        or ord(c) in PRIVATE_USE
    )


def page_status(page) -> str:
    """'ok', 'empty' (no text layer on this page) or 'bad' (text layer present but garbled)."""
    text = decode_symbol_font(page.get_text()).strip()
    if len(text) < MIN_PAGE_CHARS:
        return "empty"
    garbage = sum(is_garbage(c) for c in text)
    letters = sum(c.isalpha() or c.isspace() for c in text)
    if garbage / len(text) > MAX_GARBAGE_RATIO or letters / len(text) < MIN_LETTER_RATIO:
        return "bad"
    return "ok"


def route(doc) -> str:
    """'text' or 'ocr', decided per document so all blocks of a PDF share one shape.

    OCR when any page is garbled or no page has text (scanned). Empty pages inside an
    otherwise good PDF are usually full-page art: skipped, not worth OCR-ing the document.
    """
    statuses = [page_status(page) for page in doc]
    if "bad" in statuses or all(s == "empty" for s in statuses):
        return "ocr"
    return "text"


# =========================================================================================
# Text path (PyMuPDF)
# =========================================================================================
TEXT_BLOCK = 0      # PyMuPDF block types: 0 = text, 1 = image
MAJORITY = 0.5      # a line is bold/italic if more than this share of its characters is
# Sizes are compared in whole points: 9.0 and 9.04 are the same style, 9 and 15 are not.
SIZE_DECIMALS = 0
# Duplicate heading lines sit at the same height, give or take rounding to a whole point.
SAME_LINE_DECIMALS = 0
COORD_DECIMALS = 1  # bbox precision in the CSV, in points


def line_style(line) -> tuple:
    """(size, bold, italic) of one line, decided by the majority of its characters."""
    sizes, bold, italic, n_chars = Counter(), 0, 0, 0
    for span in line["spans"]:
        n = len(span["text"].strip())
        font = span["font"].lower()
        sizes[round(span["size"], SIZE_DECIMALS)] += n
        bold += n if (span["flags"] & pymupdf.TEXT_FONT_BOLD or "bold" in font) else 0
        italic += n if (span["flags"] & pymupdf.TEXT_FONT_ITALIC or "italic" in font) else 0
        n_chars += n
    if not n_chars:
        return (None, 0, 0)
    size = sizes.most_common(1)[0][0]
    return (size, int(bold / n_chars > MAJORITY), int(italic / n_chars > MAJORITY))


def text_blocks(doc, doc_id: str) -> list[dict]:
    """PyMuPDF blocks, split wherever the font style changes.

    PyMuPDF groups lines by position only, so a feature heading often ends up glued to the
    end of the paragraph above it. A change of size/bold/italic starts a new block, which
    puts headings in blocks of their own.
    """
    rows = []
    for page_no, page in enumerate(doc, start=1):
        # Default order follows the PDF's content stream, which keeps columns together.
        # sort=True would sort by position and interleave two-column layouts.
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != TEXT_BLOCK:
                continue

            runs, prev = [], None  # runs: [(style, [(text, bbox)])]
            for line in block["lines"]:
                text = decode_symbol_font("".join(span["text"] for span in line["spans"]))
                if not text.strip():
                    continue
                # Homebrewery draws headings twice at the same spot (fill + outline).
                _, top, _, _ = line["bbox"]
                key = (text, round(top, SAME_LINE_DECIMALS))
                if key == prev:
                    continue
                prev = key
                style = line_style(line)
                last_style, last_lines = runs[-1] if runs else (None, None)
                if last_lines is not None and last_style == style:
                    last_lines.append((text, line["bbox"]))
                else:
                    runs.append((style, [(text, line["bbox"])]))

            for (size, bold, italic), lines in runs:
                rows.append(dict(
                    doc_id=doc_id, page=page_no,
                    text="\n".join(t for t, _ in lines).strip(),
                    **_bbox_pt([b for _, b in lines]),
                    font_size=size if size is not None else "",
                    is_bold=bold, is_italic=italic,
                    layout_label="", ocr_score="", source="text",
                ))
    return rows


def _bbox_pt(boxes, scale: float = 1.0) -> dict:
    """Union of (x0, y0, x1, y1) boxes, scaled to PDF points and rounded."""
    x0s, y0s, x1s, y1s = zip(*boxes, strict=True)
    return dict(
        x0=round(min(x0s) * scale, COORD_DECIMALS),
        y0=round(min(y0s) * scale, COORD_DECIMALS),
        x1=round(max(x1s) * scale, COORD_DECIMALS),
        y1=round(max(y1s) * scale, COORD_DECIMALS),
    )


# =========================================================================================
# OCR path (PaddleOCR PP-StructureV3)
# =========================================================================================
# Render resolution: 9 pt body text becomes ~19 px tall at 150 dpi, plenty for PP-OCR, and
# a page is ~1240x1750 px. Detection then runs at that native size: no shrinking (which
# blurs small text) and no enlarging (which only costs time).
OCR_DPI = 150
OCR_DET_SIDE_LEN = 1800
POINTS_PER_INCH = 72  # PDF coordinates are in points; pixels / dpi * 72 = points

# Recognition crops are tiny; batching them is what keeps a GPU busy. On CPU, batching
# brings nothing and only raises memory use.
GPU_REC_BATCH_SIZE = 16
CPU_REC_BATCH_SIZE = 1

GPU_DEVICE = "gpu:0"  # first GPU; this machine has one
SCORE_DECIMALS = 3    # ocr_score precision in the CSV
HEIGHT_DECIMALS = 1   # font_size (line height) precision in the CSV, in points


def ocr_settings(device: str) -> dict:
    gpu = device.startswith("gpu")
    return dict(
        # 85% English accuracy vs 70% for v4 at the same size, and fewer missed spaces.
        text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        # Server detection is more accurate but ~4x slower: worth it only on a GPU.
        text_detection_model_name="PP-OCRv5_server_det" if gpu else "PP-OCRv5_mobile_det",
        text_det_limit_type="max",
        text_det_limit_side_len=OCR_DET_SIDE_LEN,
        text_recognition_batch_size=GPU_REC_BATCH_SIZE if gpu else CPU_REC_BATCH_SIZE,
        # PDF renders are upright and flat: these models would only cost time.
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        # Not needed for homebrew; dice ("1d8 + modifier") could be misread as formulas.
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_seal_recognition=False,
        # Table structure recognition loads 5 more models and is slow per table. Tables
        # will be labelled "other", so their plain text (filled in from OCR lines) is enough.
        use_table_recognition=False,
        # Kept: region detection drives multi-column reading order.
        use_region_detection=True,
        # oneDNN (mkldnn) crashes on some layout models in Paddle 3.3 on CPU
        # ("ConvertPirAttribute2RuntimeAttribute not support"). Irrelevant on GPU.
        **({} if gpu else {"enable_mkldnn": False}),
    )


@functools.lru_cache(maxsize=None)
def get_pipeline(device: str):
    # Imported lazily: text-layer PDFs never load the large OCR models, and this file
    # still runs on Pythons where PaddlePaddle isn't installable. Cached: models load once.
    from paddleocr import PPStructureV3
    return PPStructureV3(device=device, **ocr_settings(device))


def pick_device(requested: str) -> str:
    """GPU_DEVICE if Paddle can use a GPU, else 'cpu'. Warns when a GPU is wasted."""
    import paddle
    has_cuda = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    if requested == "cpu":
        return "cpu"
    if has_cuda:
        return GPU_DEVICE
    if requested == "gpu":
        sys.exit("GPU requested, but this Paddle build has no CUDA. Install paddlepaddle-gpu "
                 "(see README).")
    if not paddle.is_compiled_with_cuda():
        log("[WARN] CPU-only Paddle build: OCR will be slow. For an NVIDIA GPU install "
            "paddlepaddle-gpu (see README).")
    return "cpu"


def render(page):
    """Page -> BGR numpy image (what Paddle expects), without a temporary PNG file."""
    import numpy as np
    pix = page.get_pixmap(dpi=OCR_DPI, alpha=False)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    bgr = rgb[:, :, ::-1]  # reverse the colour channels: RGB -> BGR
    return np.ascontiguousarray(bgr)


def page_rows(data: dict, doc_id: str, page_no: int) -> list[dict]:
    """One PP-StructureV3 page result -> block rows."""
    import numpy as np
    data = data.get("res", data)  # some versions wrap the result in {"res": ...}
    to_pt = POINTS_PER_INCH / OCR_DPI  # pixels -> points, the text path's coordinates

    ocr = data.get("overall_ocr_res", {})
    lines = [
        (list(map(float, box)), text, float(score))
        # strict=False: Paddle returns equal-length lists, but a short one shouldn't crash.
        for box, text, score in zip(
            ocr.get("rec_boxes", []), ocr.get("rec_texts", []), ocr.get("rec_scores", []),
            strict=False)
    ]

    # Reading order first; blocks outside the reading flow (headers, page numbers) have
    # block_order None and go last, in detection order.
    blocks = sorted(
        data.get("parsing_res_list", []),
        key=lambda b: (b.get("block_order") is None, b.get("block_order") or 0,
                       b.get("block_id", 0)),
    )

    def center_inside(line_box, x0, y0, x1, y1) -> bool:
        lx0, ly0, lx1, ly1 = line_box
        return x0 <= (lx0 + lx1) / 2 <= x1 and y0 <= (ly0 + ly1) / 2 <= y1

    def top(line) -> float:
        (_, ly0, _, _), _, _ = line
        return ly0

    rows = []
    for block in blocks:
        x0, y0, x1, y1 = map(float, block["block_bbox"])
        inside = [line for line in lines if center_inside(line[0], x0, y0, x1, y1)]
        text = (block.get("block_content") or "").strip()
        if not text:
            # Tables (with table recognition off) and some images come without content:
            # use the OCR lines inside the block instead, top to bottom.
            text = "\n".join(t for _, t, _ in sorted(inside, key=top)).strip()
        if not text:
            continue
        heights = [ly1 - ly0 for (_, ly0, _, ly1), _, _ in inside]
        scores = [score for _, _, score in inside]
        rows.append(dict(
            doc_id=doc_id, page=page_no, text=text,
            **_bbox_pt([(x0, y0, x1, y1)], to_pt),
            font_size=(round(float(np.median(heights)) * to_pt, HEIGHT_DECIMALS)
                       if heights else ""),
            is_bold="", is_italic="",
            layout_label=block.get("block_label", ""),
            ocr_score=round(float(np.mean(scores)), SCORE_DECIMALS) if scores else "",
            source="ocr",
        ))
    return rows


def ocr_blocks(doc, doc_id: str, device: str) -> list[dict]:
    pipeline = get_pipeline(device)
    log(f"[OCR]  {doc_id} ({len(doc)} pages)")
    # All pages in one call: the pipeline batches them instead of starting cold per page.
    images = [render(page) for page in doc]
    rows = []
    for page_no, res in enumerate(pipeline.predict(input=images), start=1):
        rows += page_rows(res.json, doc_id, page_no)
    return rows


# =========================================================================================
# Driver
# =========================================================================================
def settings_stamp(source: str, device: str | None) -> dict:
    stamp = {"extractor_version": EXTRACTOR_VERSION, "source": source}
    if source == "ocr":
        stamp |= {"ocr_dpi": OCR_DPI, **ocr_settings(device)}
    return stamp


def is_up_to_date(out_meta: Path, stamp: dict) -> bool:
    try:
        return json.loads(out_meta.read_text(encoding="utf-8")) == stamp
    except (OSError, ValueError):
        return False


def write_outputs(rows: list[dict], out_csv: Path, out_meta: Path, stamp: dict) -> None:
    for idx, row in enumerate(rows):
        row["block_idx"] = idx
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    # Meta last: a crash mid-write leaves no stamp, so the PDF is redone next run.
    out_meta.write_text(json.dumps(stamp, indent=2), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="in_dir", default="media/eng/pdf")
    ap.add_argument("--pdf", action="append", default=[],
                    help="process only this PDF (repeatable); overrides --in")
    ap.add_argument("--out", dest="out_dir", default="data/blocks")
    ap.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    ap.add_argument("--only", choices=["text", "ocr"],
                    help="process only PDFs taking this path (e.g. text on Python 3.14)")
    ap.add_argument("--force", action="store_true", help="redo even if the output is up to date")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    device = None  # resolved on first OCR document, so text-only runs never import Paddle
    done, skipped, failed = 0, 0, []

    pdfs = [Path(p) for p in args.pdf] or sorted(Path(args.in_dir).glob("*.pdf"))
    for pdf in pdfs:
        out_csv = out_dir / f"{pdf.stem}.csv"
        out_meta = out_dir / f"{pdf.stem}.meta.json"
        try:
            with pymupdf.open(pdf) as doc:
                source = route(doc)
                if args.only and source != args.only:
                    continue
                if source == "ocr" and device is None:
                    device = pick_device(args.device)
                    log(f"[INFO] OCR device: {device}")

                stamp = settings_stamp(source, device)
                if not args.force and out_csv.exists() and is_up_to_date(out_meta, stamp):
                    skipped += 1
                    continue

                if source == "text":
                    log(f"[TEXT] {pdf.name}")
                    rows = text_blocks(doc, pdf.name)
                else:
                    rows = ocr_blocks(doc, pdf.name, device)

            write_outputs(rows, out_csv, out_meta, stamp)
            log(f"       -> {len(rows)} blocks")
            done += 1
        except Exception:
            # One broken PDF must not stop the batch: log it, carry on, report at the end.
            log(f"[FAIL] {pdf.name}\n{traceback.format_exc()}")
            failed.append(pdf.name)

    log(f"\nDone {done}, skipped {skipped} (up to date), failed {len(failed)}")
    for name in failed:
        log(f"  failed: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
