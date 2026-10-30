import pytest

from dnd2json.classifier.features import FEATURE_NAMES, RUN_IN_PATTERN, block_features, has_level

# Idea of this file: regexes fail silently. A wrong pattern doesn't crash anything; it just
# makes the model a bit worse and you never find out why. These tests pin down exactly
# what the patterns should and shouldn't catch.

# Real snippets from the PDFs in media/eng/pdf.
SHOULD_MATCH = [
    "Starting at 3rd level, you can",               # space
    "2nd-level Chocolatier Magic feature",          # hyphen
    "Beginning at 6th Level, your Vault",           # capital L
    "Starting at 6h level, you",                    # OCR typo
    "if you're a level 5 druid",                    # number after the word
]
SHOULD_NOT_MATCH = [
    "you gain a level of exhaustion",               # the word, no number
    "spells of a level for which you have slots",   # the word, no number
    "take 2d6 fire damage",                         # dice, not a level
    "Bards of the College of Fools use crude stories.",  # ordinary description
]


@pytest.mark.parametrize("text", SHOULD_MATCH)
def test_level_pattern_matches(text):
    assert has_level(text)


@pytest.mark.parametrize("text", SHOULD_NOT_MATCH)
def test_level_pattern_ignores(text):
    assert not has_level(text)


def test_run_in_heading():
    assert RUN_IN_PATTERN.match("**Coveted Recipe.** Also from 3rd level, after a creature")
    assert RUN_IN_PATTERN.match("Coveted Recipe. Also from 3rd level")
    assert not RUN_IN_PATTERN.match("Bards of the College of Fools use crude stories.")


def test_block_features_shape():
    assert len(block_features("At 3rd level, you gain 2d6.")) == len(FEATURE_NAMES)
    assert len(block_features("")) == len(FEATURE_NAMES)  # empty text mustn't crash
