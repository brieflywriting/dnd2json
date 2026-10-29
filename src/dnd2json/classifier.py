"""Stage 3b: the block classifier. Decides whether body text describes the subclass
(class_description) or states rules (feature).

The model is `words_only` from models.py (TF-IDF + logistic regression): it won the
comparison in evaluate.py. This module trains it on all labelled data, saves it to
data/models/, and loads it for the pipeline.

Why it's needed when the layout already shows headings: a heading only says "a new
section starts here", not what kind. The classifier tells an intro or a lore sidebar
("Roleplaying a Fool") from a feature, and catches documents whose features have no
level line and no "At 3rd level" phrase.
"""

import datetime
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

from dnd2json import config
from dnd2json.clean import normalize_text
from dnd2json.errors import ModelNotTrainedError
from dnd2json.models import make_words_only

LABELS = ["class_description", "feature"]  # what the model predicts; "other" is rule-based
DESCRIPTION = "class_description"


def load_training_data() -> pd.DataFrame:
    """All labelled blocks the model may learn from: real (OCR) blocks + SRD blocks.

    Blocks labelled `other` (see LABELS.md) are left out: the model only separates
    descriptions from features; page furniture is handled by rules in roles.py.
    Texts go through the same normalisation as pipeline blocks, so the model sees
    training text in the same shape as the text it will classify.
    """
    files = [config.GOLD_DIR / name for name in config.GOLD_REAL_FILES]
    files.append(config.GOLD_DIR / config.GOLD_SRD_FILE)
    frames = []
    for path in files:
        if not path.exists():
            warnings.warn(f"training file missing, skipped: {path}", stacklevel=2)
            continue
        df = pd.read_csv(path, encoding="utf-8").dropna(subset=["text", "label"])
        frames.append(df[["text", "label"]])
    if not frames:
        raise FileNotFoundError(f"no training data found in {config.GOLD_DIR}")
    data = pd.concat(frames, ignore_index=True)
    data = data[data["label"].isin(LABELS)].copy()
    data["text"] = data["text"].map(normalize_text)
    return data.reset_index(drop=True)


def train(save_to=config.MODEL_PATH) -> dict:
    """Fit the model on all training data and save it. Returns the saved metadata."""
    data = load_training_data()
    model = make_words_only()
    model.fit(data[["text"]], data["label"])
    meta = {
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "rows": len(data),
        "label_counts": data["label"].value_counts().to_dict(),
    }
    save_to.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "meta": meta}, save_to)
    return meta


class BlockClassifier:
    """The trained model, loaded once and reused for every document."""

    def __init__(self, path=config.MODEL_PATH):
        if not path.exists():
            raise ModelNotTrainedError(
                f"no trained classifier at {path}. Run `dnd2json train` first.")
        try:
            saved = joblib.load(path)
            self.model, self.meta = saved["model"], saved["meta"]
        except Exception as e:  # corrupt file, pickled with an incompatible version...
            raise ModelNotTrainedError(
                f"couldn't load {path} ({e}). Retrain with `dnd2json train`.") from e
        if self.meta.get("sklearn_version") != sklearn.__version__:
            warnings.warn(
                f"classifier was trained with scikit-learn {self.meta.get('sklearn_version')}"
                f", running {sklearn.__version__}. Retrain if predictions look off.",
                stacklevel=2)
        classes = list(self.model.classes_)
        self._description_column = classes.index(DESCRIPTION)

    def description_probability(self, texts) -> np.ndarray:
        """P(class_description) for each text; 1 - that is P(feature)."""
        texts = list(texts)
        if not texts:
            return np.array([])
        probabilities = self.model.predict_proba(pd.DataFrame({"text": texts}))
        return probabilities[:, self._description_column]
