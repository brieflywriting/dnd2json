"""Step 4.2: hand-made features for one block of text."""

import re

import numpy as np
import pandas as pd

# TODO 4.2: the level pattern.
# Idea: features announce when you get them ("Starting at 3rd level"). Descriptions almost
#       never do, so this is one of the strongest single clues.
# Hints:
#   - Write down every way your documents spell it: with a space, with a hyphen,
#     capitalised, with the number after the word "level".
#   - "a level of exhaustion" has the word but no number, so it must not match.
#   - Two alternatives joined by "or", and ignore case.
LEVEL_PATTERN = None

# TODO 4.2: the run-in heading pattern.
# Idea: homebrew often starts a paragraph with a bold name and a period
#       ("**Coveted Recipe.** Also..."). That's a feature heading glued to its body.
# Hints:
#   - Anchor it to the very start of the text.
#   - A few capitalised words, then a period, with optional bold markers around them.
#   - Test it against an ordinary sentence that also starts with a capital letter.
#     It must not match.
RUN_IN_PATTERN = None

# TODO 4.2: the dice pattern.
# Idea: dice notation (1d8, 2d6, d20) only appears in mechanics.
# Hints:
#   - The leading number is optional; the d and the trailing number are not.
#   - Use word boundaries so "and8" or "d" inside words don't count.
DICE_PATTERN = None

# TODO 4.2: word sets for "who is the text talking to/about".
# Idea: features address the player ("you"); descriptions talk about the subclass's people
#       ("they", "members"). Read your blocks and collect the words that show it.
SECOND_PERSON = set()
THIRD_PERSON = set()

# TODO 4.2: names for the numbers.
# Idea: one name per number that block_features returns, in the same order. You'll want
#       them when you look at which features the model relies on, and the test uses the
#       count.
FEATURE_NAMES = []


def has_level(text: str) -> bool:
    # TODO 4.2
    # Idea: a yes/no question about one block. Used by the rules and by the features, so
    #       it lives in one place.
    # Hints:
    #   - "Does the pattern occur anywhere?" vs "does it occur at the start?" are
    #     different regex methods. Which one do you want?
    raise NotImplementedError


def block_features(text: str) -> list[float]:
    """One block -> a fixed-length list of numbers."""
    # TODO 4.2
    # Idea: describe the *shape* of a block with numbers a model can weigh: how long, how
    #       many lines, how mechanical, who it talks to. Every block must give the same
    #       numbers in the same order.
    # Hints:
    #   - Split the text once at the top (into lines, lowercase words, letters) and reuse
    #     the pieces.
    #   - Use fractions instead of counts where length would distort them ("you" per word,
    #     not "you" total).
    #   - Some blocks are tiny. Make every division safe when the bottom is zero.
    #   - Ideas from the plan: length, line count, ends with a period, digit fraction,
    #     capital fraction, level, run-in heading, you-rate, they-rate, short-line
    #     fraction (tables), dice count.
    #   - Skip "relative position in doc": your CSV rows are sorted by label, so position
    #     would leak the answer. It needs a real block_idx column first.
    raise NotImplementedError


def hand_features(X: pd.DataFrame) -> np.ndarray:
    """DataFrame with a `text` column -> 2-D array, one row per block."""
    # TODO 4.2
    # Idea: the bridge between "one block" and "a whole table" that a sklearn pipeline
    #       can call.
    # Hints:
    #   - Run block_features on every text in the table.
    #   - sklearn wants a 2-D numeric array back: rows = blocks, columns = features.
    raise NotImplementedError
