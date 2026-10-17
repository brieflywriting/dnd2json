import pytest

from dnd2json.features import FEATURE_NAMES, RUN_IN_PATTERN, block_features, has_level

# TODO 4.2: fill both lists.
# Hint: should match: "Starting at 3rd level", "14th-level feature", "6th Level", "level 6 spell"
# Hint: should not match: "a level of exhaustion", "3d6 damage", a normal description sentence
SHOULD_MATCH = []
SHOULD_NOT_MATCH = []


@pytest.mark.parametrize("text", SHOULD_MATCH)
def test_level_pattern_matches(text):
    # TODO 4.2
    # Hint: assert has_level(text)
    raise NotImplementedError


@pytest.mark.parametrize("text", SHOULD_NOT_MATCH)
def test_level_pattern_ignores(text):
    # TODO 4.2
    # Hint: assert not has_level(text)
    raise NotImplementedError


def test_run_in_heading():
    # TODO 4.2
    # Hint: assert RUN_IN_PATTERN.match("**Coveted Recipe.** Also, you...")
    #       and assert not RUN_IN_PATTERN.match("The need for wizards in any army is great.")
    raise NotImplementedError


def test_block_features_shape():
    # TODO 4.2
    # Hint: len(block_features("You hit it. It takes 1d8 damage.")) == len(FEATURE_NAMES)
    raise NotImplementedError
