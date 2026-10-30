"""Every model we compare. Each takes a DataFrame with a `text` column and predicts a label.

The models form a ladder. Each step adds one thing, so evaluate.py can show what that
one thing is worth:

    dummy             nothing                    -> the floor
    rules             your intuition, as ifs     -> how far common sense gets
    hand_lr           hand features, learned     -> are learned weights better than ifs?
    hand+words        + word TF-IDF              -> does vocabulary add anything?
    hand+words+chars  + character TF-IDF         -> does robustness to OCR breaks help?
    words_only        word TF-IDF alone          -> were the hand features worth writing?

All learned models share one classifier (_logreg), so they differ only in their features.
"""

import re

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from dnd2json.classifier.features import (
    DICE_PATTERN,
    SECOND_PERSON,
    THIRD_PERSON,
    hand_features,
    has_level,
)


def make_dummy():
    # The dumbest possible "model": it ignores the text and always says the most common
    #   label. Its score is the floor; a model that can't beat it has learned nothing.
    #
    # Why DummyClassifier: sklearn ships it for exactly this, and it behaves like any
    #   other model (fit/predict), so the harness needs no special case.
    # Why "most_frequent": it's the strongest no-information guess for accuracy (70%),
    #   which makes the point of macro-F1 visible: accuracy 0.70, macro-F1 0.41, because
    #   the description class gets F1 = 0.
    #   Other strategies exist ("stratified" guesses at random in the label ratio,
    #   "uniform" 50/50). Normally: use "most_frequent" as the baseline for
    #   classification; it's the one people expect to see.
    return DummyClassifier(strategy="most_frequent")


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


# ---------------------------------------------------------------------------------------
# Building blocks for the learned models
# ---------------------------------------------------------------------------------------
#
# Every learned model is: ColumnTransformer (turns the table into numbers) -> _logreg().
#
# Why ColumnTransformer? It routes columns of a table into transformers and glues their
#   outputs side by side into one matrix. That's how hand features (11 numbers) and
#   TF-IDF (thousands of word weights) end up as one input for one classifier. It also
#   means every model accepts the same input: the one-column table cross_validate passes.
#   Normally: use it whenever different columns, or different views of one column, need
#   different preprocessing.
#
# The column selector decides the SHAPE each transformer receives:
#   ["text"] (a list)   -> a one-column DataFrame. hand_features reads X["text"] by name,
#                          so it needs this.
#   "text"   (a string) -> a 1-D Series of strings. TfidfVectorizer needs this: given a
#                          DataFrame, it would iterate over the column NAMES and see one
#                          "document" called "text", silently learning nothing.
#   Normally: list for transformers that want a table, string for ones that want a
#   plain sequence (text vectorizers are the main example).
#
# The entries are built by functions, not module-level constants: every model, and every
#   fold, must get brand-new unfitted transformers. A shared transformer object would
#   carry its fitted vocabulary/scaling from one fold into the next.


def _hand():
    # Hand features as a pipeline step: text table -> 11 numbers -> standardised.
    #
    # FunctionTransformer(hand_features): wraps a plain function as a pipeline step, so it
    #   gets fit/transform like everything else. It has nothing to learn; fit does nothing.
    #   Normally: use it for stateless preprocessing (feature formulas, log transforms).
    #   If the step needs to learn something from training data, write a real transformer
    #   class instead.
    #
    # StandardScaler: rescales each feature to mean 0, spread 1. Logistic regression's
    #   regularisation punishes large weights. n_chars is in the hundreds and needs a
    #   tiny weight; you_rate is around 0.1 and needs a huge weight to matter, so
    #   without scaling the penalty unfairly squashes the small-range features.
    #   Normally: scale numeric features for any linear model, SVM, k-NN or neural net.
    #   Tree models (random forests, gradient boosting) don't need it.
    #
    # Being inside the pipeline matters: the scaler learns its mean and spread in fit(),
    #   which cross_validate only calls on the training rows. Scaling the whole dataset
    #   up front would let test rows influence the training: a small but real leak.
    return make_pipeline(FunctionTransformer(hand_features), StandardScaler())


def _words():
    # Word TF-IDF: each block becomes a weight per word (and word pair).
    #   TF  = how often the word appears in this block.
    #   IDF = down-weights words that appear in every block ("the", "a"), so rarer,
    #         more telling words ("saving", "college") carry more weight.
    #   Each row is then normalised to length 1, so a long block isn't "louder" than a
    #   short one.
    #
    # sublinear_tf=True: uses 1 + log(count) instead of the raw count. A feature that
    #   says "spell" eight times isn't eight times more of a feature than one that says
    #   it once. Normally: a sensible default for documents longer than a sentence.
    #
    # ngram_range=(1, 2): single words plus pairs of neighbouring words. Pairs catch
    #   phrases whose meaning is more than their parts: "saving throw", "bonus action",
    #   "college of", "you can". Normally: (1, 2) is the usual step up from words only;
    #   beyond 2 the number of columns explodes and most n-grams appear once.
    #
    # min_df=2: drop words/pairs that appear in only one block. With ~200 training
    #   blocks, a term seen once can't generalise; it can only memorise that one block.
    #   Normally: raise min_df as data grows, or use it as a fraction (e.g. 0.01).
    return TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)


def _chars():
    # Character TF-IDF: each block becomes weights for short character chunks.
    #
    # Why: OCR breaks words ("C\nhaotic", "draws upon.the"). Word TF-IDF sees "haotic" as
    #   a brand-new word it has never seen. Character chunks like "hao", "aot", "otic"
    #   still match the unbroken parts, so broken words keep most of their signal.
    #
    # analyzer="char_wb": chunks are taken inside word boundaries (each word padded with
    #   spaces), instead of "char", which also makes chunks spanning two words.
    #   Cross-word chunks ("u ca" from "you can") are mostly noise and multiply the
    #   column count. Normally: char_wb for robustness to typos/OCR, char when spacing
    #   itself is unreliable (e.g. languages without spaces).
    #
    # ngram_range=(2, 5): 2-character chunks catch prefixes/suffixes, 5 is about the size
    #   of a short word or a meaningful stem ("spell", "level"). Beyond 5 the chunks start
    #   behaving like whole words again and lose the OCR robustness.
    #
    # sublinear_tf and min_df: same reasoning as _words().
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)


def _logreg():
    # The one classifier every learned model shares, so the models differ only in their
    #   features and the comparison is fair.
    #
    # Why logistic regression:
    #   - Strong on text: thousands of sparse TF-IDF columns, few rows. That's the
    #     classic setting where linear models match or beat fancier ones.
    #   - Readable: each feature gets one weight; positive pushes towards one label,
    #     negative towards the other. You can print the top words and check they make
    #     sense (we did this in legacy/train_first_try.py: "you", "spell" vs "they", "college").
    #   - Fast, and gives probabilities, which step 6.1 needs for the "unknown" threshold.
    #   Alternatives: LinearSVC (similar accuracy, no probabilities out of the box), naive
    #   Bayes (fast, weaker when features are correlated, as ours are), random forests /
    #   gradient boosting (poor on sparse text, good on the hand features alone).
    #   Normally: start with logistic regression for text classification and only move
    #   on if the error analysis shows it can't capture something.
    #
    # class_weight="balanced": mistakes on the rare label (class_description, ~30%) cost
    #   more, in proportion to how rare it is. Without it, the model leans towards always
    #   saying "feature", which is exactly what macro-F1 punishes. Normally: use it
    #   whenever classes are imbalanced and you care about the rare one.
    #
    # max_iter=1000: the optimiser stops after 100 steps by default and warns that it
    #   didn't converge. A larger limit lets it finish; it stops early once it has.
    #
    # C (regularisation strength) stays at the default 1.0 on purpose. Tuning it is
    #   legitimate, but only inside the training folds (e.g. LogisticRegressionCV). Picking
    #   C by looking at evaluate.py's scores would be tuning on the test set.
    return LogisticRegression(class_weight="balanced", max_iter=1000)


def _pipeline(*entries):
    # Shared shape of every learned model: route the text into the given feature
    #   blocks, glue them side by side, classify. Writing it once means each make_
    #   function only lists WHICH features it uses, and the four models can't drift apart
    #   by accident (e.g. one forgetting class_weight).
    #
    # Each entry is (name, transformer, column selector). The names only matter for
    #   inspecting the fitted model later, e.g. model[0].named_transformers_["words"].
    return make_pipeline(ColumnTransformer(list(entries)), _logreg())


# ---------------------------------------------------------------------------------------
# The learned models
# ---------------------------------------------------------------------------------------


def make_hand_lr():
    # Step 4.3: only your 11 hand features, weighed by a learned model instead of ifs.
    #   If it beats rules, learned weights use your intuition better than you did by hand.
    #   If it doesn't, the features themselves are the limit, not how they're combined.
    return _pipeline(("hand", _hand(), ["text"]))


def make_hand_words():
    # Step 4.4: hand features + vocabulary. Words like "saving throw" or "College of"
    #   carry meaning the 11 numbers can't.
    #   Note the two selectors: ["text"] for hand features (wants a table), "text" for
    #   TF-IDF (wants plain strings). See the block comment above.
    return _pipeline(
        ("hand", _hand(), ["text"]),
        ("words", _words(), "text"),
    )


def make_hand_words_chars():
    # Step 4.4: + character chunks, for robustness to OCR-broken words.
    #   If this doesn't beat hand+words, your OCR is clean enough that whole words suffice,
    #   and the extra columns aren't worth their cost.
    return _pipeline(
        ("hand", _hand(), ["text"]),
        ("words", _words(), "text"),
        ("chars", _chars(), "text"),
    )


def make_words_only():
    # Extra comparison: your first model (legacy/train_first_try.py), but in the harness.
    #   Compare with hand+words: if the gap is smaller than the fold std, the hand
    #   features add nothing and you could drop them (less code, nothing lost).
    return _pipeline(("words", _words(), "text"))


# The names evaluate.py accepts on the command line. Order = order they run in.
MODELS = {
    "dummy": make_dummy,
    "rules": make_rules,
    "hand_lr": make_hand_lr,
    "hand+words": make_hand_words,
    "hand+words+chars": make_hand_words_chars,
    "words_only": make_words_only,
}
