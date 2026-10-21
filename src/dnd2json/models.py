"""Every model we compare. Each takes a DataFrame with a `text` column and predicts a label."""

import re

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from dnd2json.features import DICE_PATTERN, SECOND_PERSON, THIRD_PERSON, hand_features, has_level


def make_dummy():
    # TODO 3.2
    # Idea: the dumbest possible "model": it ignores the text and always says the most
    #       common label. Its score is the floor. A real model that can't beat it has
    #       learned nothing.
    # Hints:
    #   - scikit-learn ships a classifier made exactly for this; it has a strategy option.
    #   - Compare its accuracy with its macro-F1. Why are they so different?
    return DummyClassifier(strategy="most_frequent")
    #raise NotImplementedError


class RulesClassifier(ClassifierMixin, BaseEstimator):
    """Step 4.1: a few hand-written ifs, shaped like a sklearn model so the harness can run it."""

    # Idea of the class: write down what *you* would check by eye. If a few ifs already do
    # well, a learned model has to beat your intuition, not just the dummy. Making it look
    # like a sklearn model (fit + predict) means the harness needs no special case for it.

    # Why inherit from ClassifierMixin and BaseEstimator?
    #   - BaseEstimator gives get_params()/set_params(). sklearn tools such as clone(),
    #     GridSearchCV and cross_val_score copy models through those, so any custom model
    #     you might hand to sklearn should inherit it.
    #   - ClassifierMixin adds .score() (accuracy) and marks the class as a classifier, so
    #     tools like StratifiedKFold-based helpers treat it correctly.
    #   - Order matters: mixins go first, BaseEstimator last. That's sklearn's own rule.
    #   Normally: use both whenever you write your own model class. Our cross_validate
    #   doesn't strictly need them, but they cost nothing and keep the door open.

    def fit(self, X, y):
        # Rules learn nothing, so fit() only records which labels exist.
        #
        # self.classes_: every sklearn classifier sets this during fit(). The trailing
        #   underscore is sklearn's convention for "learned from data, exists only after
        #   fit()". sklearn uses it, for example, to check a model is fitted, and
        #   predict_proba() columns follow its order. np.unique() returns the labels
        #   sorted, which is what sklearn's own classifiers do.
        #   Normally: always set it in a custom classifier's fit(), taken from y, never
        #   hard-coded. A hard-coded list goes stale the moment you add a label (and a
        #   typo in it, like "self_description", would never be noticed).
        self.classes_ = np.unique(y)

        # return self: lets you chain `model = make_rules().fit(X, y)`. Every sklearn
        #   estimator does this, and Pipeline relies on it.
        #   Normally: always, in every fit() you write.
        return self

    def predict(self, X):
        # X is the one-column table cross_validate passes in. X["text"] takes the column
        #   out; looping over the table itself would loop over its column *names*
        #   (just "text"), not its rows.
        #   Normally: select the column by name, so the model also works if the table
        #   later gets more columns (source, doc_id, ...).
        #
        # np.array(...): sklearn models return numpy arrays, and cross_validate and the
        #   metrics expect that. Returning a plain list would mostly work, but matching
        #   the convention avoids surprises (e.g. y_pred[mask] needs an array).
        #
        # A list comprehension with a Python function is fine here: 134 blocks, simple
        #   checks. Normally: for large data or numeric work you'd use vectorised
        #   pandas/numpy operations instead of a per-row Python loop.
        return np.array([self._rule(text) for text in X["text"]])

    @staticmethod
    def _rule(text):
        # @staticmethod: the rule doesn't need `self` (it reads no settings and stores
        #   nothing), so it's a plain function that lives in the class for tidiness.
        #   Normally: use it for helpers like this. If the rule ever gets a setting
        #   (e.g. a threshold you want to tune), make it a normal method and store the
        #   setting in __init__ instead.
        #
        # The leading underscore in _rule means "internal, not part of the public API".
        #   Python doesn't enforce it; it's a note to readers.
        #
        # Check order matters: the most reliable check goes first, because the first
        # check that fires wins and the later ones never see the block.

        # Check 1: game mechanics. A level ("3rd-level") or dice ("2d6") almost never
        #   appears in a description, so when either is present we can say "feature" with
        #   very few false alarms. It only fires on about half of the features, which is
        #   fine: the next checks handle the rest.
        if has_level(text) or DICE_PATTERN.search(text):
            return "feature"

        # Check 2: who is the text about? Features talk to the player ("you can..."),
        #   descriptions talk about the subclass's people ("Bards of this college... they").
        #   - lower() so "Your" and "your" count the same.
        #   - Curly apostrophes from OCR are straightened so "you’re" matches "you're".
        #   - The regex keeps letters and apostrophes, which drops punctuation: "you."
        #     becomes "you".
        #   Counting with sum(word in SET ...) works because True counts as 1. Sets make
        #   each `in` check instant, which is why the word lists are sets, not lists.
        words = re.findall(r"[a-z']+", text.lower().replace("’", "'"))
        n_you = sum(word in SECOND_PERSON for word in words)
        n_they = sum(word in THIRD_PERSON for word in words)

        # Plain comparison, no tuned margin. ">=" or a ratio scored about a point
        #   higher on the data, but that's inside the fold-to-fold noise, and picking the
        #   threshold by looking at the score is tuning on the test set. A rules baseline
        #   is meant to be "simple intuition", so keep it simple.
        #   Known miss: second-person descriptions ("You are an Influencer...") end up
        #   with more "you" than "they" and fall through to "feature". No simple rule
        #   catches those without breaking others; that's a job for a learned model.
        if n_they > n_you:
            return "class_description"

        # Check 3: fallback. When nothing fired, guess the more common label (about 70%
        #   of blocks are features), so an unsure guess is right more often than not.
        #   Normally: a rules model always needs a default, otherwise some inputs get no
        #   answer (here, the function would return None).
        return "feature"


def make_rules():
    # A fresh, unfitted instance every call. cross_validate calls this once per fold so
    #   no fold reuses a model fitted on another fold's data. For rules nothing is learned,
    #   but keeping the same "make_" shape means evaluate.py treats every model the same.
    return RulesClassifier()


def _hand():
    # TODO 4.3
    # Idea: turn text into your hand-made numbers, then put all the numbers on a
    #       comparable scale.
    # Hints:
    #   - sklearn has a wrapper that turns a plain function into a pipeline step.
    #   - Length is in the hundreds, fractions are between 0 and 1. Regularization
    #     punishes big weights, so unscaled small-range features get unfairly ignored.
    #     Which preprocessing step fixes that?
    # Template:
    #   return make_pipeline(<hand_features wrapped as a step>, <a scaler>)
    # Think: the scaler learns a mean and spread in fit(). Which rows does it learn them
    #        from inside cross_validate, and why is that the honest choice?
    raise NotImplementedError


def _logreg():
    # TODO 4.3
    # Idea: one shared classifier definition, so all models differ only in their features.
    # Hints:
    #   - You have about 70% of one label and 30% of the other. There's an option that
    #     makes mistakes on the rare class cost more.
    #   - Give the optimizer enough iterations to finish.
    # Template:  return LogisticRegression(<balance option>, <iterations>)
    raise NotImplementedError


def make_hand_lr():
    # TODO 4.3
    # Idea: the first learned model. It sees only your hand features, so its score tells
    #       you how much your intuition is worth once a model weighs it properly.
    # Hints:
    #   - Use the tool that sends chosen columns of a table into chosen transformers.
    #     Here: the text column into _hand().
    #   - Your feature function reads the text column *by name*, so it must receive a
    #     table, not a bare column. How you name the column decides which one it gets.
    #   - Chain that with _logreg().
    # Template:
    #   features = ColumnTransformer([
    #       (<a name>, <transformer>, <column selector>),
    #   ])
    #   return make_pipeline(features, _logreg())
    # Think: the selector "text" and the selector ["text"] hand the transformer different
    #        shapes. Which one does hand_features need, and which one does TF-IDF need?
    #        (Try both in a shell on a 3-row DataFrame and print what comes out.)
    raise NotImplementedError


def make_hand_words():
    # TODO 4.4
    # Idea: add vocabulary. Words like "saving throw" or "College of" carry meaning your
    #       hand features can't. Both kinds of features sit side by side in one input.
    # Hints:
    #   - Same column-routing tool as before, with a second entry for word TF-IDF.
    #   - TF-IDF wants a plain list of strings, the opposite of what _hand() wants.
    #   - Long blocks repeat words a lot. There's an option that dampens raw counts.
    # Template: same as make_hand_lr, plus one more entry:
    #       (<name>, TfidfVectorizer(<dampen option>), <selector>)
    # Think: three of make_hand_lr / make_hand_words / make_hand_words_chars /
    #        make_words_only now build almost the same thing. Is a small helper worth it?
    raise NotImplementedError


def make_hand_words_chars():
    # TODO 4.4
    # Idea: OCR breaks words ("C\nhaotic", "you�re"), so whole-word matching misses them.
    #       Short character chunks still match the unbroken parts.
    # Hints:
    #   - A third entry: TF-IDF again, but over characters inside word boundaries,
    #     chunks of 2 to 5 characters.
    # Template: make_hand_words + one entry:
    #       (<name>, TfidfVectorizer(analyzer=<?>, ngram_range=<?>, ...), <selector>)
    # Think: "char" vs "char_wb": print the chunks each makes from "you're a bard"
    #        (vectorizer.build_analyzer()("...")). Which one suits broken OCR words?
    raise NotImplementedError


def make_words_only():
    # TODO 4.4 (extra comparison)
    # Idea: your train.py model (TF-IDF + logistic regression) without hand features. If
    #       it scores the same as hand+words, your hand features add nothing.
    # Hints:
    #   - It still has to accept the same table input as every other model.
    # Template: a ColumnTransformer with only the word TF-IDF entry, then _logreg().
    # Think: when you compare this with hand+words, how big must the gap be (look at the
    #        fold std) before you believe the hand features help?
    raise NotImplementedError


# TODO: uncomment each line once its function works, so evaluate.py can run it.
MODELS = {
     "dummy": make_dummy,
    "rules": make_rules,
    # "hand_lr": make_hand_lr,
    # "hand+words": make_hand_words,
    # "hand+words+chars": make_hand_words_chars,
    # "words_only": make_words_only,
}
