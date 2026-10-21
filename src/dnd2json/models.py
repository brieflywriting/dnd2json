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

    def fit(self, X, y):
        self.classes_=["self_description", "feature"]
        # TODO 4.1
        # Idea: rules learn nothing from data, but the harness calls fit() on every model.
        # Hints:
        #   - sklearn classifiers remember which labels exist after fitting. Store that.
        #   - What does fit() return, so that "model = make_x().fit(...)" works?
        # Template:
        #   self.classes_ = <every distinct label in y, sorted>
        #   return <?>
        raise NotImplementedError

    def predict(self, X):
        # TODO 4.1
        # Idea: apply the single-block rule to every block in the table.
        # Hints:
        #   - X is a table; your rule wants one string at a time.
        #   - Return the same kind of array other sklearn models return.
        # Template:
        #   labels = [<the rule applied to t> for t in <the text column of X>]
        #   return <labels as a numpy array>
        # Think: why does looping over X itself (not its column) give the wrong thing?
        raise NotImplementedError

    @staticmethod
    def _rule(text):
        # TODO 4.1
        # Idea: one block of text in, one label out, decided by a few checks in order.
        #       The order matters: the most reliable check goes first.
        # Hints:
        #   - Before writing anything, read ~10 blocks of each label side by side.
        #   - Who does a feature talk to, and who does a description talk about?
        #   - What only appears in game mechanics (levels, dice, ...)? features.py helps here.
        #   - When nothing fires, which label is the safer guess?
        # Template (three checks, most reliable first):
        #   1. if <a mechanics signal: level? dice?>:
        #          return <label>
        #   2. words  = <lowercase words of text>       (block_features in features.py does this)
        #      n_you  = <how many words are in SECOND_PERSON>
        #      n_they = <how many words are in THIRD_PERSON>
        #      if <your comparison of n_you and n_they>:
        #          return <label>
        #   3. return <fallback label>
        # Think:
        #   - Step 2: plain "n_they > n_you", or a margin/ratio? Try one, run evaluate,
        #     look at the confusion matrix, change it ONCE.
        #   - "You are an Influencer..." is a description in second person. Will any
        #     simple rule catch it? Should it even try?
        #   - Swap steps 1 and 2. Does the score change? Why?
        # You'll need: import re, and SECOND_PERSON / THIRD_PERSON from dnd2json.features.
        raise NotImplementedError


def make_rules():
    # TODO 4.1
    # Idea: every model is built by a "make_" function, so the harness can ask for a
    #       fresh one in every fold.
    # Template:  return <a new instance of the class>
    raise NotImplementedError


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
    # "rules": make_rules,
    # "hand_lr": make_hand_lr,
    # "hand+words": make_hand_words,
    # "hand+words+chars": make_hand_words_chars,
    # "words_only": make_words_only,
}
