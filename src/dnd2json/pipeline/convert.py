"""Run the whole pipeline: PDF -> blocks -> clean -> roles -> classify -> structure -> JSON.

    convert_pdf(pdf)     one PDF -> a validated Document (raises on hard errors)
    convert_many(pdfs)   a batch: never stops on one bad PDF, writes JSON + a run report

Stage by stage (each module's docstring explains its stage in detail):
    1. extract.py     PDF -> data/blocks/<name>.csv  (text layer here, OCR via .venv-ocr)
    2. clean.py       reading order, junk, drop caps, text repair
    3. roles.py       heading / body / table / level_line / other
       ../classifier/block_classifier.py   body text -> P(description)
    4. structure.py   sections -> subclasses, description, features
    5. schema.py      validated JSON
"""

import json
import subprocess
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf

from dnd2json import config
from dnd2json.classifier.block_classifier import BlockClassifier
from dnd2json.errors import (
    DnD2JsonError,
    ExtractionError,
    NoBlocksError,
    OCRUnavailableError,
    StructureError,
)
from dnd2json.pipeline import clean, extract, roles, structure
from dnd2json.pipeline.schema import Document

# Mean OCR confidence below this gets a warning: the text likely has many misreads.
LOW_OCR_CONFIDENCE = 0.85
# OCR of one PDF should take seconds on a GPU; this only stops a hung process.
OCR_TIMEOUT_S = 15 * 60
# Columns written to the debug CSV, in this order.
DEBUG_COLUMNS = ["block_idx", "page", "role", "role_reason", "p_desc", "font_size",
                 "is_bold", "layout_label", "ocr_score", "text"]


# =========================================================================================
# Stage 1: make sure the blocks CSV exists
# =========================================================================================
def ensure_blocks(pdf: Path, blocks_dir: Path = config.BLOCKS_DIR, force: bool = False) -> Path:
    """Return the blocks CSV for `pdf`, extracting it first if needed.

    Text-layer PDFs are extracted right here. Scanned/garbled PDFs need PaddleOCR, which
    only installs on Python 3.12, so extract.py is run with the OCR venv's interpreter.
    An existing CSV is reused when it was made by the current extractor version.
    """
    out_csv = blocks_dir / f"{pdf.stem}.csv"
    out_meta = blocks_dir / f"{pdf.stem}.meta.json"
    try:
        with pymupdf.open(pdf) as doc:
            source = extract.route(doc)
            if not force and _blocks_current(out_csv, out_meta, source):
                return out_csv
            if source == "text":
                rows = extract.text_blocks(doc, pdf.name)
                extract.write_outputs(rows, out_csv, out_meta, extract.settings_stamp("text", None))
                return out_csv
    except (pymupdf.FileDataError, RuntimeError, OSError) as e:
        raise ExtractionError(f"can't read {pdf.name}: {e}") from e

    _run_ocr(pdf, blocks_dir, force)
    if not out_csv.exists():
        raise ExtractionError(f"OCR finished but wrote no blocks for {pdf.name}")
    return out_csv


def _blocks_current(out_csv: Path, out_meta: Path, source: str) -> bool:
    """Up to date = made by this extractor version, down the same path (text/ocr).

    OCR settings differ between GPU and CPU runs; either is fine, so they're not compared.
    """
    if not (out_csv.exists() and out_meta.exists()):
        return False
    try:
        meta = json.loads(out_meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (meta.get("extractor_version") == extract.EXTRACTOR_VERSION
            and meta.get("source") == source)


def _run_ocr(pdf: Path, blocks_dir: Path, force: bool) -> None:
    if not config.OCR_PYTHON.exists():
        raise OCRUnavailableError(
            f"{pdf.name} needs OCR, but the OCR environment isn't set up "
            f"({config.OCR_PYTHON} missing). See README, 'OCR setup'.")
    command = [str(config.OCR_PYTHON), str(config.EXTRACT_SCRIPT),
               "--pdf", str(pdf), "--out", str(blocks_dir)]
    if force:
        command.append("--force")
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=OCR_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        raise ExtractionError(f"OCR of {pdf.name} timed out after {OCR_TIMEOUT_S}s") from e
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-15:])
        raise ExtractionError(f"OCR of {pdf.name} failed:\n{tail}")


# =========================================================================================
# Stages 2-5 for one PDF
# =========================================================================================
def convert_pdf(pdf: Path, classifier: BlockClassifier, *, force_extract: bool = False,
                debug_dir: Path | None = None) -> Document:
    """One PDF -> Document. Raises a DnD2JsonError subclass on hard errors."""
    pdf = Path(pdf)
    blocks_csv = ensure_blocks(pdf, force=force_extract)

    df = clean.load_blocks(blocks_csv)
    extraction = str(df["source"].iloc[0])
    df, stats = clean.clean_blocks(df)
    df = roles.assign_roles(df, stats["body_font_size"])

    # Only body text is classified; other roles are decided by layout.
    df["p_desc"] = np.nan
    body = df["role"] == "body"
    if body.any():
        df.loc[body, "p_desc"] = classifier.description_probability(df.loc[body, "text"])

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        df[DEBUG_COLUMNS].to_csv(debug_dir / f"{pdf.stem}.debug.csv", index=False,
                                 encoding="utf-8")

    if not body.any() and not (df["role"] == "table").any():
        raise NoBlocksError("no body text left after cleaning")

    subclasses, warnings = structure.build_subclasses(df, pdf.stem)
    if not subclasses or not any(s.description or s.features for s in subclasses):
        raise StructureError("found neither a description nor any feature")

    if extraction == "ocr":
        confidence = df["ocr_score"].mean()
        if pd.notna(confidence) and confidence < LOW_OCR_CONFIDENCE:
            warnings.append(f"low OCR confidence ({confidence:.2f}): expect misread words")

    return Document(source_file=pdf.name, extraction=extraction,
                    subclasses=subclasses, warnings=warnings)


# =========================================================================================
# Batch
# =========================================================================================
def convert_many(pdfs, out_dir: Path = config.OUTPUT_DIR, *, force_extract: bool = False,
                 debug: bool = False) -> list[dict]:
    """Convert several PDFs. One failure never stops the batch.

    Writes <out_dir>/<name>.json per PDF and <out_dir>/_report.json summarising the run.
    Returns the report rows (one dict per PDF).
    """
    classifier = BlockClassifier()  # load once; raises ModelNotTrainedError if missing
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug" if debug else None
    report = []

    for pdf in pdfs:
        pdf = Path(pdf)
        row = {"file": pdf.name, "status": "ok"}
        try:
            document = convert_pdf(pdf, classifier, force_extract=force_extract,
                                   debug_dir=debug_dir)
            (out_dir / f"{pdf.stem}.json").write_text(document.to_json(), encoding="utf-8")
            row.update(
                extraction=document.extraction,
                subclasses=[s.name for s in document.subclasses],
                features=sum(len(s.features) for s in document.subclasses),
                warnings=document.warnings,
            )
            if document.warnings:
                row["status"] = "warnings"
        except DnD2JsonError as e:
            # Expected failure modes (unreadable PDF, no OCR env, nothing found).
            row.update(status="failed", error=f"{type(e).__name__}: {e}")
        except Exception as e:  # noqa: BLE001 - a bug; keep going, but record the traceback
            row.update(status="error", error=f"{type(e).__name__}: {e}",
                       traceback=traceback.format_exc())
        report.append(row)
        _print_row(row)

    (out_dir / "_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    return report


def _print_row(row: dict) -> None:
    status = row["status"].upper()
    if row["status"] in ("ok", "warnings"):
        names = ", ".join(row["subclasses"])
        print(f"[{status:8s}] {row['file']}: {names} ({row['features']} features)", flush=True)
        for warning in row.get("warnings", []):
            print(f"             - {warning}", flush=True)
    else:
        print(f"[{status:8s}] {row['file']}: {row['error']}", flush=True)
