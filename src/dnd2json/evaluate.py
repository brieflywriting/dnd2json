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

# TODO 3.1: the folder that holds your CSVs.
# Idea: a path that works no matter which folder you run the script (or pytest) from.
# Hint: "./data" is relative to where you are standing when you run it, not to this file.
#       Path objects know where the current file lives; walk up from there to the repo root.
DATA = Path(__file__)


def load_data():
    """Return (real, srd): real blocks are scored on, SRD blocks are extra training data only."""


    pathToSRD = DATA.parent/"data"/"gold"/"5e-SRD-Combined.csv"
    pathToOCR1 = DATA.parent/"data"/"gold"/"5e-OCR-Combined.csv")
    pathToOCR2=Path("./data/gold/5e-OCR-CombinedNewClasses.csv")

    OCR_data1=pd.read_csv(pathToOCR1).dropna(subset=["text"])
   # OCR_data2=pd.read_csv(pathToOCR2).dropna(subset=["text"])
    SRD_data=pd.read_csv(pathToSRD).dropna(subset=["text"])

   # OCR_maindata=pd.concat([OCR_data1,OCR_data2])
    
    return OCR_data1.reset_index(drop=True),SRD_data.reset_index(drop=True)
    # TODO 3.1a
    # Idea: real and SRD data play different roles. Real blocks are what you care about, so
    #       they are the only thing you ever score on. SRD blocks are clean extra examples
    #       that may only ever help training. Keep them as two separate tables so the
    #       rest of the code can't mix them up.
    # Hints:
    #   - Which files hold your labelled real blocks and your labelled SRD blocks? Look in
    #     data/gold. Is any file there one you promised not to look at until step 5.3?
    #   - Some rows have no text. A model can't learn from those, so drop them.
    #   - Your data contract (1.3) says every row knows where it came from.
    #   - Later code expects both tables to have the same columns. SRD has no document
    #     id. What could you put there?
    #   - The splitter later hands you row positions. After dropping rows, do the
    #     positions still line up with the index?
   # raise NotImplementedError


def cross_validate(make_model, real, srd, n_splits=5, seed=42):
    """Return (y_pred, fold_scores): one out-of-fold prediction per real block + macro-F1 per fold."""

    cv=StratifiedGroupKFold(n_splits,shuffle=True,random_state=seed)
    y_pred=np.empty(len(real), dtype=object)
    y_true=real["label"].to_numpy()
    fold_scores=[]
    for train_idx, test_idx in cv.split(real, real["label"], groups=real["doc_id"]):
       model= make_model()
       train_real=real.iloc[train_idx]
       test_real=real.iloc[test_idx]

       train_data = pd.concat([train_real, srd], ignore_index=True)

       X_train=train_data[["text"]]
       Y_train=train_data["label"]
       X_test=test_real[["text"]]

       model.fit(X_train,Y_train)
       predictions=np.asarray(model.predict(X_test)).reshape(-1)

       if len(predictions)!= len(test_idx):
            raise ValueError(
                f"Expected {len(test_idx)} predictions, "
                f"got {len(predictions)}."
            )
       y_pred[test_idx]=predictions


       fold_scores.append(f1_score(y_true[test_idx],predictions,average="macro",zero_division=0))
    return y_pred, fold_scores
    # TODO 3.1b
    # Idea: simulate "the model meets a document it has never seen", several times, so
    #       every real block gets tested exactly once. Blocks from one document share an
    #       author's wording, so a document must never be in train and test at the same time.
    #       Otherwise the model can recognise the author instead of learning the label.
    # Hints:
    #   - The splitter needs three things: the rows, the labels (to keep the class balance
    #     similar in every fold) and the groups (what must stay together).
    #   - Every model should get exactly the same folds, or one model could get easier
    #     folds by luck. What makes a shuffle repeatable?
    #   - Each split gives you two lists of row positions. Which rows go to training, and
    #     which source of data is only ever allowed in training?
    #   - Build a brand-new model in every fold. Why would reusing last fold's model be
    #     cheating?
    #   - Decide one input shape for all models (a one-column table, not a bare column of
    #     strings), because the model in step 4.4 needs to look columns up by name.
    #   - Make one container the size of `real` before the loop. Each fold fills in only
    #     its own test positions; after the last fold every position is filled.
    #   - Also keep a macro-F1 score per fold. You'll want to see how much it jumps around.
   # raise NotImplementedError


def report(name, y_true, y_pred, fold_scores):
    """Print classification_report + confusion matrix + macro-F1. Return macro-F1."""
    # TODO 3.1c
    # Idea: one number hides *how* a model is wrong. Show three views:
    #       - per-class precision/recall: which label does it struggle with?
    #       - the confusion matrix: what does it mistake that label for?
    #       - macro-F1: the single number you compare models by. It weights both classes
    #         equally, so always guessing the common class scores badly.
    #       Plus the spread across folds, so you know whether two scores really differ.
    # Hints:
    #   - A model that never predicts a class makes precision 0/0. The report function has
    #     a setting for what to show instead of a warning.
    #   - A bare confusion matrix is just numbers. Give it the label names on rows and
    #     columns, and fix the label order so every model's matrix reads the same way.
    #   - Average and standard deviation of the fold scores. If two models are closer than
    #     the spread, treat them as a tie.
    #   - Return the macro-F1 so main() can build its summary.
    raise NotImplementedError


def main(argv=None):
    # TODO 3.1d
    # Idea: one command runs every model through the identical test and ends with a side-by-
    #       side comparison. This is what you'll rerun after every change in step 4.
    # Hints:
    #   - Let the command line name which models to run. If it names none, run them all.
    #     Where does Python keep command-line words?
    #   - Load the data once, outside the model loop.
    #   - For each model: cross-validate, report, and remember its score.
    #   - Finish with a short table: one line per model, its name and its macro-F1.
    raise NotImplementedError


if __name__ == "__main__":
    main()
