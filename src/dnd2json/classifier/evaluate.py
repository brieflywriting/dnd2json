"""Step 3: one fair test for every model.

Run:  dnd2json evaluate                (all models)
      dnd2json evaluate dummy rules    (only these)
  or  python -m dnd2json.classifier.evaluate [models...]
"""

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from dnd2json import config
from dnd2json.classifier.models import MODELS

DATA = config.GOLD_DIR

LABELS = ["class_description", "feature"]

# Every model is cross-validated once per seed. One seed's folds can be lucky or unlucky;
# averaging over several shows whether a gap between two models is real or noise.
SEEDS = [42, 0, 1, 2, 3]


def load_data():
    """Return (real, srd): real blocks are scored on, SRD blocks are extra training data only."""
    real = pd.read_csv(DATA / "5e-OCR-Combined.csv").dropna(subset=["text"])
    srd = pd.read_csv(DATA / "5e-SRD-Combined.csv").dropna(subset=["text"])

    # Data contract (1.3): every row knows where it came from.
    real["source"] = "real"
    srd["source"] = "srd"

    return real.reset_index(drop=True), srd.reset_index(drop=True)


def cross_validate(make_model, real, srd, n_splits=5, seed=42):
    """Return (y_pred, fold_scores): one out-of-fold prediction per real block, and
    macro-F1 per fold."""
    cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    y_pred = np.empty(len(real), dtype=object)
    y_true = real["label"].to_numpy()
    fold_scores = []

    for train_idx, test_idx in cv.split(real, real["label"], groups=real["doc_id"]):
        model = make_model()
        train_data = pd.concat([real.iloc[train_idx], srd], ignore_index=True)
        test_real = real.iloc[test_idx]

        model.fit(train_data[["text"]], train_data["label"])
        predictions = np.asarray(model.predict(test_real[["text"]])).reshape(-1)

        if len(predictions) != len(test_idx):
            raise ValueError(f"Expected {len(test_idx)} predictions, got {len(predictions)}.")
        y_pred[test_idx] = predictions

        fold_scores.append(
            f1_score(y_true[test_idx], predictions, average="macro", zero_division=0))

    return y_pred, fold_scores


def report(name, y_true, y_pred, fold_scores):
    """Print classification_report + confusion matrix + macro-F1. Return macro-F1."""
    print(f"=== {name} ===")
    print(classification_report(y_true, y_pred, zero_division=0))

    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    print("Confusion matrix (rows = true, columns = predicted):")
    print(pd.DataFrame(matrix, index=LABELS, columns=LABELS))

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    n_errors = int((np.asarray(y_true) != np.asarray(y_pred)).sum())
    print(
        f"Macro-F1 {macro_f1:.3f}   errors {n_errors}/{len(y_true)}   "
        f"folds {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f} "
        f"({len(fold_scores)} folds over {len(SEEDS)} seeds)"
    )
    return macro_f1, n_errors


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]          # words typed after the command; [0] is the script itself

    names = argv or list(MODELS)     # nothing typed -> run every model

    unknown = [name for name in names if name not in MODELS]
    if unknown:
        print(f"Unknown model(s): {', '.join(unknown)}")
        print(f"Choose from: {', '.join(MODELS)}")
        return

    real, srd = load_data()          # once, so every model sees the same data

    results = {}
    for name in names:
        # The detailed report shows the first seed's predictions; the fold spread pools
        # every seed, so it reflects both fold luck and split luck.
        all_fold_scores = []
        for seed in SEEDS:
            y_pred_seed, fold_scores = cross_validate(MODELS[name], real, srd, seed=seed)
            all_fold_scores += fold_scores
            if seed == SEEDS[0]:
                y_pred = y_pred_seed
        macro_f1, n_errors = report(name, real["label"], y_pred, all_fold_scores)
        results[name] = (macro_f1, n_errors, np.mean(all_fold_scores), np.std(all_fold_scores))
        print()

    print(f"Summary ({len(real)} real blocks):")
    print(f"  {'model':20s} {'macro-F1':>8s} {'errors':>7s} {'folds mean ± std':>18s}")
    ranked = sorted(results.items(), key=lambda kv: kv[1][0], reverse=True)
    for name, (macro_f1, n_errors, mean, std) in ranked:
        print(f"  {name:20s} {macro_f1:8.3f} {n_errors:7d} {mean:11.3f} ± {std:.3f}")


if __name__ == "__main__":
    main()
