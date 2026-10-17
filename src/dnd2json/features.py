"""Step 4.2: hand-made features for one block of text."""

import re

import numpy as np
import pandas as pd

# TODO 4.2: replace None with re.compile(...).
# Hint: should match "3rd level", "14th-level", "6th Level", "level 6";
#       should NOT match "a level of exhaustion". Two alternatives joined with |:
#       \d{1,2}(st|nd|rd|th)[\s-]level   and   level\s\d{1,2}   + re.IGNORECASE
LEVEL_PATTERN = None

# TODO 4.2: "**Coveted Recipe.** Also..." or "Coveted Recipe. Also..."
# Hint: from the start (^): optional \*\*, then 1-6 Title Case words, a period, optional \*\*,
#       then whitespace. Make sure a normal sentence like "The need for wizards ..." fails.
RUN_IN_PATTERN = None

# TODO 4.2: dice like "1d8", "2d6", "d20".
# Hint: \b\d*d\d+\b
DICE_PATTERN = None

# TODO 4.2: word sets for the "you" vs "they" rates.
# Hint: {"you", "your", "yours", "yourself"} and {"they", "their", "them", "members", ...}
SECOND_PERSON = set()
THIRD_PERSON = set()

# TODO 4.2: one name per number that block_features returns, in the same order.
# Hint: handy for debugging; the pytest shape check compares lengths.
FEATURE_NAMES = []


def has_level(text: str) -> bool:
    # TODO 4.2
    # Hint: LEVEL_PATTERN.search(text) is not None
    raise NotImplementedError


def block_features(text: str) -> list[float]:
    """One block -> a fixed-length list of numbers."""
    # TODO 4.2
    # Hint: compute once at the top: lines (non-empty), words (lowercase), letters.
    # Hint: guard every division with max(..., 1): some blocks are tiny.
    # Hint: candidates: n_chars, n_lines, ends_with_period, digit_frac, capital_frac,
    #       has_level, has_run_in, you_frac, they_frac, short_line_frac (< 25 chars), n_dice.
    # Hint: SKIP "relative position in doc": your CSV rows are sorted by label, so position
    #       would leak the answer. It needs a real block_idx column first.
    raise NotImplementedError


def hand_features(X: pd.DataFrame) -> np.ndarray:
    """DataFrame with a `text` column -> 2-D array, one row per block."""
    # TODO 4.2
    # Hint: np.array([block_features(t) for t in X["text"]])
    raise NotImplementedError
