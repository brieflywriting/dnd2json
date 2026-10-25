"""Sanity checks for the labelled CSVs in data/gold.

Hand-edited data breaks silently: a wrong encoding, a typo in a label or a missing doc_id
won't crash training, it just makes every score a little wrong. Run `pytest` after every
edit to the CSVs.
"""

from pathlib import Path

import pandas as pd
import pytest

GOLD = Path(__file__).resolve().parents[1] / "data" / "gold"
LABELS = {"class_description", "feature"}
REQUIRED_COLUMNS = {"text", "label", "doc_id"}

CSV_FILES = sorted(GOLD.glob("*.csv"))
# SRD repeats a few features word for word across classes ("Extra Attack", "Epic Boon").
# That's real, so SRD only has to label its duplicates consistently.
REAL_FILES = [path for path in CSV_FILES if "SRD" not in path.name]


def read(path):
    return pd.read_csv(path, encoding="utf-8")


def test_gold_files_exist():
    assert CSV_FILES, f"No CSV files in {GOLD}"


@pytest.mark.parametrize("path", CSV_FILES, ids=lambda p: p.name)
def test_is_utf8(path):
    path.read_bytes().decode("utf-8")  # raises UnicodeDecodeError if the file is cp1252 etc.


@pytest.mark.parametrize("path", CSV_FILES, ids=lambda p: p.name)
def test_no_replacement_characters(path):
    # U+FFFD "�" means a character was destroyed by a wrong encoding round-trip.
    # It can't be recovered from the CSV; fix the text from the source PDF.
    damaged = read(path)["text"].fillna("").str.contains("�")
    assert not damaged.any(), f"Rows with '�': {damaged[damaged].index.tolist()}"


@pytest.mark.parametrize("path", CSV_FILES, ids=lambda p: p.name)
def test_required_columns(path):
    missing = REQUIRED_COLUMNS - set(read(path).columns)
    assert not missing, f"Missing columns: {missing}"


@pytest.mark.parametrize("path", CSV_FILES, ids=lambda p: p.name)
def test_no_empty_cells(path):
    df = read(path)
    for column in REQUIRED_COLUMNS:
        empty = df[column].isna() | (df[column].astype(str).str.strip() == "")
        assert not empty.any(), f"Empty '{column}' in rows {df.index[empty].tolist()}"


@pytest.mark.parametrize("path", CSV_FILES, ids=lambda p: p.name)
def test_labels_are_known(path):
    unknown = set(read(path)["label"].dropna()) - LABELS
    assert not unknown, f"Unknown labels: {unknown}"


@pytest.mark.parametrize("path", REAL_FILES, ids=lambda p: p.name)
def test_no_duplicate_texts(path):
    df = read(path)
    duplicated = df["text"].str.strip().duplicated(keep=False)
    assert not duplicated.any(), f"Duplicate texts in rows {df.index[duplicated].tolist()}"


@pytest.mark.parametrize("path", CSV_FILES, ids=lambda p: p.name)
def test_duplicates_agree_on_label(path):
    df = read(path)
    labels_per_text = df.groupby(df["text"].str.strip())["label"].nunique()
    conflicting = labels_per_text[labels_per_text > 1]
    assert conflicting.empty, f"Same text, different labels: {conflicting.index.str[:60].tolist()}"


def test_no_text_shared_between_files():
    # The same block in two files can end up in both training and test data.
    seen = {}
    for path in CSV_FILES:
        for text in read(path)["text"].dropna().str.strip():
            seen.setdefault(text, set()).add(path.name)
    shared = {text[:60]: files for text, files in seen.items() if len(files) > 1}
    assert not shared, f"Texts in more than one file: {shared}"
