import pytest

from dnd2json.features import FEATURE_NAMES, RUN_IN_PATTERN, block_features, has_level

# Idea of this file: regexes fail silently. A wrong pattern doesn't crash anything; it just
# makes the model a bit worse and you never find out why. These tests pin down exactly
# what the patterns should and shouldn't catch.

# TODO 4.2: fill both lists.
# Hints:
#   - SHOULD_MATCH: every way your documents write a level (spacing, hyphen,
#     capitalisation, number before or after the word).
#   - SHOULD_NOT_MATCH: near-misses. The word "level" without a number, dice notation,
#     an ordinary description sentence.
#   - Copy real snippets from your data where you can.
SHOULD_MATCH = []
SHOULD_NOT_MATCH = []


@pytest.mark.parametrize("text", SHOULD_MATCH)
def test_level_pattern_matches(text):
    # TODO 4.2
    # Idea: every positive example must be caught.
    raise NotImplementedError


@pytest.mark.parametrize("text", SHOULD_NOT_MATCH)
def test_level_pattern_ignores(text):
    # TODO 4.2
    # Idea: no near-miss may be caught (no false alarms).
    raise NotImplementedError


def test_run_in_heading():
    # TODO 4.2
    # Idea: one bold-heading example must match and one ordinary sentence must not.
    #       Checking both sides stops a pattern that matches everything from passing.
    raise NotImplementedError


def test_block_features_shape():
    # TODO 4.2
    # Idea: the number of features must equal the number of names, so the two lists
    #       can't silently drift apart when you add a feature.
    raise NotImplementedError
