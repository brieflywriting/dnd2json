"""Paths and settings shared by the pipeline, in one place.

Every other module imports its paths from here, so moving a folder means changing one line.
All paths are absolute (built from this file's location), so the code works no matter
which folder you run it from.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/dnd2json/config.py -> repo root

# --- Inputs -----------------------------------------------------------------------------
PDF_DIR = ROOT / "media" / "eng" / "pdf"     # the homebrew PDFs
GOLD_DIR = ROOT / "data" / "gold"            # hand-labelled blocks (classifier training)

# --- Intermediate and final outputs -------------------------------------------------------
BLOCKS_DIR = ROOT / "data" / "blocks"        # stage 1 output: one CSV of blocks per PDF
OUTPUT_DIR = ROOT / "data" / "output"        # stage 5 output: one JSON per PDF
EXPECTED_DIR = ROOT / "data" / "expected"    # hand-written correct JSON for `check`
MODEL_PATH = ROOT / "data" / "models" / "block_classifier.joblib"

# --- OCR environment ------------------------------------------------------------------------
# PaddlePaddle can't be installed on Python 3.14, so OCR runs in a separate Python 3.12
# venv (see README). The pipeline calls extract.py with that interpreter for scanned PDFs.
_VENV_PYTHON = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
OCR_PYTHON = ROOT / ".venv-ocr" / _VENV_PYTHON
EXTRACT_SCRIPT = ROOT / "src" / "dnd2json" / "extract.py"

# --- Gold training data ---------------------------------------------------------------------
# Labelled real blocks. The SRD file is added separately (clean extra examples).
GOLD_REAL_FILES = ["5e-OCR-Combined.csv", "5e-OCR-CombinedNewClasses.csv"]
GOLD_SRD_FILE = "5e-SRD-Combined.csv"
