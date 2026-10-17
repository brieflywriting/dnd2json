"""Step 3: one fair test for every model.

Run:  python -m dnd2json.evaluate              (all models)
      python -m dnd2json.evaluate dummy rules  (only these)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from dnd2json.models import MODELS

DATA = Path(__file__).resolve().parents[2] / "data" / "gold"


def load_data():
    """Return (real, srd): real blocks are scored on, SRD blocks are extra training data only."""


    pathToSRD = Path("./data/srd/5e-SRD-Subclasses.json")
    pathToOCR1 = Path("./data/srd/5e-SRD-Features.json")
    pathToOCR2=Path("./data/gold/5e-OCR-CombinedNewClasses.csv")

    OCR_data1=pd.read_csv(pathToOCR1).dropna(subset=["text"])
    OCR_data2=pd.read_csv(pathToOCR2).dropna(subset=["text"])
    SRD_data=pd.read_csv(pathToSRD).dropna(subset=["text"])

    
    return pd.concat(OCR_data1,OCR_data2,SRD_data).reset_index(drop=True)
    # TODO 3.1a
    # Hint: read DATA / "5e-OCR-Combined.csv" and DATA / "5e-SRD-Combined.csv".
    
    # Hint: .dropna(subset=["text"]) on both (the SRD file has an empty row).
    # Hint: add a "source" column ("real" / "srd"); give SRD a fake doc_id = "srd"
    #       so both tables have the same columns.
    # Hint: .reset_index(drop=True) so row positions match the indices cv.split returns.
    raise NotImplementedError


def cross_validate(make_model, real, srd, n_splits=5, seed=42):
    """Return (y_pred, fold_scores): one out-of-fold prediction per real block + macro-F1 per fold."""

    cv=StratifiedGroupKFold(n_splits,shuffle=True,random_state=seed)

    for i in cv.split(real, real["label"], groups=real["doc_id"]):
        
    # TODO 3.1b
    # Hint: cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    #       -> same seed = same splits for every model = fair comparison.
    # Hint: loop over cv.split(real, real["label"], groups=real["doc_id"]).
    # Hint: train = real train rows + ALL of srd; test = real test rows only.
    # Hint: make_model() gives a fresh, untrained model every fold.
    # Hint: always pass train[["text"]] (double brackets = DataFrame), so every model
    #       gets the same kind of input (step 4.4 needs a DataFrame).
    # Hint: y_pred = np.empty(len(real), dtype=object) before the loop;
    #       fill y_pred[test_idx] inside it.
    raise NotImplementedError


def report(name, y_true, y_pred, fold_scores):
    """Print classification_report + confusion matrix + macro-F1. Return macro-F1."""
    # TODO 3.1c
    # Hint: classification_report(..., zero_division=0) silences the 0/0 warning from dummy.
    # Hint: confusion_matrix(y_true, y_pred, labels=labels) -> wrap in
    #       pd.DataFrame(cm, index=labels, columns=labels) so rows/columns are named.
    # Hint: also print np.mean / np.std of fold_scores: the std tells you how big a
    #       difference between two models has to be before it means anything.
    raise NotImplementedError


def main(argv=None):
    # TODO 3.1d
    # Hint: names = argv or sys.argv[1:] or every key in MODELS.
    # Hint: load once, then for each name: cross_validate(MODELS[name], ...) and report(...).
    # Hint: finish with a summary table: one line per model with its macro-F1.
    raise NotImplementedError


if __name__ == "__main__":
    main()
