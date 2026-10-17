"""Every model we compare. Each takes a DataFrame with a `text` column and predicts a label."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from dnd2json.features import DICE_PATTERN, hand_features, has_level


def make_dummy():
    # TODO 3.2
    # Hint: DummyClassifier(strategy="most_frequent"). Expect ~0.70 accuracy but ~0.41 macro-F1.
    raise NotImplementedError


class RulesClassifier(ClassifierMixin, BaseEstimator):
    """Step 4.1: a few hand-written ifs, shaped like a sklearn model so the harness can run it."""

    def fit(self, X, y):
        # TODO 4.1
        # Hint: learns nothing. Set self.classes_ = np.unique(y) and return self.
        raise NotImplementedError

    def predict(self, X):
        # TODO 4.1
        # Hint: np.array([self._rule(t) for t in X["text"]])
        raise NotImplementedError

    @staticmethod
    def _rule(text):
        # TODO 4.1
        # Hint: read ~10 blocks of each label first, then write the ifs. Ideas:
        #   level pattern or dice (has_level, DICE_PATTERN)        -> "feature"
        #   first word is you / your / when / starting ...         -> "feature"
        #   "they" / "their" / "members" in the first ~40 words    -> "class_description"
        #   otherwise                                              -> "feature" (more common)
        raise NotImplementedError


def make_rules():
    # TODO 4.1
    # Hint: just return RulesClassifier().
    raise NotImplementedError


def _hand():
    # TODO 4.3
    # Hint: make_pipeline(FunctionTransformer(hand_features), StandardScaler())
    #       Scaling matters: n_chars is in the hundreds, the fractions are 0..1.
    raise NotImplementedError


def _logreg():
    # TODO 4.3
    # Hint: LogisticRegression(class_weight="balanced", max_iter=1000)
    raise NotImplementedError


def make_hand_lr():
    # TODO 4.3
    # Hint: make_pipeline(ColumnTransformer([("hand", _hand(), ["text"])]), _logreg())
    #       ["text"] as a LIST -> your function receives a DataFrame and reads X["text"].
    raise NotImplementedError


def make_hand_words():
    # TODO 4.4
    # Hint: same as make_hand_lr, plus ("words", TfidfVectorizer(sublinear_tf=True), "text")
    #       "text" as a STRING -> TfidfVectorizer gets one column of strings.
    raise NotImplementedError


def make_hand_words_chars():
    # TODO 4.4
    # Hint: make_hand_words plus
    #       ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
    #        "text")  -> character chunks survive OCR damage like "C\nhaotic" or "you�re".
    raise NotImplementedError


def make_words_only():
    # TODO 4.4 (extra comparison)
    # Hint: the model from your train.py, but with the input wrapped:
    #       ColumnTransformer([("words", TfidfVectorizer(), "text")]) + _logreg().
    #       Answers "do my hand features add anything over words alone?"
    raise NotImplementedError


# TODO: uncomment each line once its function works, so evaluate.py can run it.
MODELS = {
    # "dummy": make_dummy,
    # "rules": make_rules,
    # "hand_lr": make_hand_lr,
    # "hand+words": make_hand_words,
    # "hand+words+chars": make_hand_words_chars,
    # "words_only": make_words_only,
}
