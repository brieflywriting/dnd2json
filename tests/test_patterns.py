"""The wording rules in patterns.py, checked against real snippets from the PDFs.

When a new PDF breaks a pattern: add its text here first (the test fails), then fix the
pattern until every case passes. That way an old case can't silently break.
"""

import pytest

from dnd2json import patterns
from dnd2json.clean import repair_ocr


@pytest.mark.parametrize("text, level", [
    ("3rd-level College of Fools feature", 3),
    ("2nd-level Chocolatier Magic feature", 2),
    ("14th-level College of Drama feature", 14),
    ("6th-level Spycraft feature.", 6),
])
def test_level_line(text, level):
    match = patterns.LEVEL_LINE.match(text)
    assert match and int(match.group(1)) == level


@pytest.mark.parametrize("text", [
    "At 3rd level, your magic can strike a chord",       # a sentence, not a level line
    "3rd-level spells you cast with this feature deal",  # rules text
])
def test_level_line_ignores_sentences(text):
    assert not patterns.LEVEL_LINE.match(text)


def test_leading_level_line_splits_only_before_a_new_sentence():
    glued = "3rd-level College of Liturgies feature You learn the thaumaturgy cantrip"
    match = patterns.LEADING_LEVEL_LINE.match(glued)
    assert match and match.group().strip() == "3rd-level College of Liturgies feature"
    assert not patterns.LEADING_LEVEL_LINE.match("3rd-level spells you cast with this feature deal")


@pytest.mark.parametrize("text, level", [
    ("At 3rd level, your magic can strike a chord", 3),
    ("Beginning when you select this school at 2nd level, the gold", 2),
    ("...defend yourself from it. Beginning at level 3, you may calculate", 3),
    ("Also at 6ht level, the dark energies you study", 6),
    ("From 14th level, you gain the ability", 14),
])
def test_level_grant(text, level):
    match = patterns.LEVEL_GRANT.search(text) or patterns.LEVEL_IN_TEXT.search(text)
    assert match and patterns.level_number(match) == level


def test_level_grant_ignores_spell_levels():
    assert not patterns.LEVEL_IN_TEXT.search("cast a bard spell of 1st level or higher")


@pytest.mark.parametrize("text", [
    "Homebrew by Wyll Hood",
    "Art by Olga Drebas https://www.artstation.com/olgadrebas",
    "Art Credit : Robson Michael, Spike Chunsoft",
    "A wizard subclass Cover rart by Lizzart",
    "All art created by me, u/Yorviing",
    "This is unofficial Fan Content permitted under the Fan Content Policy.",
    "DD HOMEBREW",
])
def test_credits(text):
    assert patterns.CREDITS.search(text)


@pytest.mark.parametrize("text", [
    "checks made to discern the truth of your statements are made with disadvantage.",
    "the creature created by your Soul Reawaken feature lasts for up to 8 hours",
    "You are a true artist of the stage.",
])
def test_credits_ignore_rules_text(text):
    assert not patterns.CREDITS.search(text)


@pytest.mark.parametrize("text", [
    "SubcLAss FEATures", "UPDATED CLASS FEATURES", "ROGUISH ARCHETYPES", "Arcane Traditions",
    "Rogue Archetype", "14 CHAPTER 2 | SUBCLASSES", "A wizard subclass", "Fleshwarping Features",
])
def test_section_header(text):
    assert patterns.SECTION_HEADER.match(text)


@pytest.mark.parametrize("text", ["Chaotic Rift", "Bard Level College Feature", "Extra Attack"])
def test_section_header_ignores_feature_names(text):
    assert not patterns.SECTION_HEADER.match(text)


@pytest.mark.parametrize("raw, fixed", [
    ("At 1Oth level", "At 10th level"),
    ("within 6oft of you", "within 60ft of you"),
    ("1 4th-level College of Fools feature", "14th-level College of Fools feature"),
])
def test_ocr_repairs(raw, fixed):
    assert repair_ocr(raw) == fixed
