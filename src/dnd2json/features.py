"""Step 4.2: hand-made features for one block of text."""

import re

import numpy as np
import pandas as pd


#this contains mainly regex made by a clanker+ feature seekers such as common phrases

# Features announce when you get them. Covers "3rd level", "3rd-level", "6th Level",
# the OCR typo "6h level", and "level 3". "a level of exhaustion" has no number -> no match.
LEVEL_PATTERN = re.compile(r"\b\d+(?:st|nd|rd|th|h)?[\s-]+level\b|\blevel\s+\d+\b", re.IGNORECASE)

# A feature heading glued to its body: "**Coveted Recipe.** Also..." or "Coveted Recipe. Also...".
# Up to five capitalised words, then a period, at the very start of the block.
RUN_IN_PATTERN = re.compile(r"^\s*(?:\*\*)?(?:[A-Z][\w'’-]*\s+){0,4}[A-Z][\w'’-]*\.(?:\*\*)?\s")

# Dice notation: 1d8, 2d6, d20. Only appears in mechanics.
DICE_PATTERN = re.compile(r"\b\d*d\d+\b", re.IGNORECASE)

# Features talk to the player, descriptions talk about the subclass's people.
SECOND_PERSON = {"you", "your", "yours", "yourself", "you're", "you've", "you'll", "you'd"}
THIRD_PERSON = {"they", "their", "them", "theirs", "themselves", "they're", "they've", "who"}

FEATURE_NAMES = [
    "n_chars",
    "n_lines",
    "ends_with_period",
    "digit_frac",
    "upper_frac",
    "has_level",
    "has_run_in",
    "you_rate",
    "they_rate",
    "short_line_frac",
    "dice_count",
]


def has_level(text: str) -> bool:
    # search, not match: the level can be anywhere in the block, not only at the start.
    return LEVEL_PATTERN.search(text) is not None


def _safe_div(top: float, bottom: float) -> float:
    return top / bottom if bottom else 0.0


def block_features(text: str) -> list[float]:
    """One block -> a fixed-length list of numbers."""
    stripped = text.strip()
    lines = [line for line in stripped.splitlines() if line.strip()]
    # OCR uses curly apostrophes; straighten them so "you’re" counts as "you're".
    words = re.findall(r"[a-z']+", stripped.lower().replace("’", "'"))
    letters = [ch for ch in stripped if ch.isalpha()]

    return [
        len(stripped),
        len(lines),
        float(stripped.endswith(".")),
        _safe_div(sum(ch.isdigit() for ch in stripped), len(stripped)),
        _safe_div(sum(ch.isupper() for ch in letters), len(letters)),
        float(has_level(stripped)),
        float(RUN_IN_PATTERN.match(stripped) is not None),
        _safe_div(sum(w in SECOND_PERSON for w in words), len(words)),
        _safe_div(sum(w in THIRD_PERSON for w in words), len(words)),
        # Tables are many short lines. 25 chars is shorter than a normal wrapped line in your PDFs.
        _safe_div(sum(len(line.strip()) < 25 for line in lines), len(lines)),
        len(DICE_PATTERN.findall(stripped)),
    ]


def hand_features(X: pd.DataFrame) -> np.ndarray:
    """DataFrame with a `text` column -> 2-D array, one row per block."""
    return np.array([block_features(text) for text in X["text"]], dtype=float)
