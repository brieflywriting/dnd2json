"""Text patterns used by the cleaning, role and structure stages.

Everything that "knows" about D&D homebrew wording lives here, so when a new PDF breaks a
rule, this is usually the file to change. Every pattern lists real examples from the PDFs
in media/eng/pdf; tests/test_patterns.py checks them.
"""

import re

# =========================================================================================
# Classes and subclass names
# =========================================================================================
CLASS_NAMES = [
    "Artificer", "Barbarian", "Bard", "Cleric", "Druid", "Fighter", "Monk", "Paladin",
    "Ranger", "Rogue", "Sorcerer", "Warlock", "Wizard",
]

# Words a subclass title starts with, and the class each one implies.
# "College of Chaos" -> Bard, "Roguish Archetype: Mad Dog" -> Rogue, "Oath of Hope" -> Paladin.
SUBCLASS_PREFIX_TO_CLASS = {
    "college of": "Bard",
    "school of": "Wizard",
    "schools of": "Wizard",
    "arcane tradition": "Wizard",
    "roguish archetype": "Rogue",
    "rogue archetype": "Rogue",
    "martial archetype": "Fighter",
    "oath of": "Paladin",
    "circle of": "Druid",
    "path of the": "Barbarian",
    "path of": "Barbarian",
    "way of": "Monk",
    "primal path": "Barbarian",
    "sorcerous origin": "Sorcerer",
    "otherworldly patron": "Warlock",
    "divine domain": "Cleric",
    "ranger conclave": "Ranger",
    "artificer specialist": "Artificer",
}

# A heading that names a subclass: "COLLEGE OF FOOLS", "School of Hemomancy",
# "Giruza's SCHOOL OF MYTHOS" (an author's possessive may come first).
SUBCLASS_TITLE = re.compile(
    r"^(?:[\w’']+'s\s+)?(?:" + "|".join(re.escape(p) for p in SUBCLASS_PREFIX_TO_CLASS) + r")\b",
    re.IGNORECASE,
)

# Class prefix in front of a title: "Bard: College of Chivalry", "Wizard - School of...",
# "Roguish Archetype: Mad Dog", "Arcane Tradition: Fleshwarping", "Wizard:Spycraft".
TITLE_PREFIX = re.compile(
    r"^(?:(?:" + "|".join(CLASS_NAMES) + r")|roguish archetype|rogue archetype|"
    r"arcane tradition|martial archetype)\s*[:\-–]\s*",
    re.IGNORECASE,
)

# PDF filenames: "Bard - College of Chaos.pdf", "Wizard – Chocolatier.pdf",
# "Bard - College of Chaos_2.pdf" (a second version of the same subclass).
FILENAME = re.compile(r"^(?P<cls>[A-Za-z]+)\s*[-–]\s*(?P<name>.+?)(?:_\d+)?$")

# =========================================================================================
# Levels
# =========================================================================================
_ORDINAL = r"(\d{1,2})\s*(?:st|nd|rd|th|ht|h)"  # "3rd", "14th", OCR typos "6h", "6ht"
# A level in either word order: "3rd level" / "3rd-level" (group 1) or "level 3" (group 2).
_LEVEL = rf"(?:{_ORDINAL}[\s-]+level\b|level\s+(\d{{1,2}})\b)"

# A line that only announces the level, common in GM Binder / Homebrewery templates:
# "3rd-level College of Fools feature", "2nd-level Chocolatier Magic feature".
LEVEL_LINE = re.compile(rf"^\s*{_ORDINAL}[\s-]+level\b.{{0,70}}\bfeatures?\s*\.?\s*$",
                        re.IGNORECASE)

# The same line glued to the start of the paragraph (text-layer PDFs, same font):
# "3rd-level College of Liturgies feature You learn the thaumaturgy cantrip...".
# The paragraph must continue with a capital letter (a new sentence), so rules text such
# as "3rd-level spells you cast with this feature deal..." isn't split.
LEADING_LEVEL_LINE = re.compile(
    rf"^\s*{_ORDINAL}[\s-]+level\b[^.\n]{{0,70}}?\bfeatures?\b\s*(?=[A-Z\"“'])")

# Level stated inside the rules text: "At 3rd level, ...", "Starting at 6th level",
# "Beginning when you select this school at 2nd level", "From 14th level", "Also at 3rd
# level", "When you reach 6th level", "Beginning at level 3". Only phrases that grant the
# feature: "a spell of 1st level" doesn't match, because "of" isn't a lead-in word.
# The level is in group 1 or group 2 (see _LEVEL); use level_number() to read it.
LEVEL_IN_TEXT = re.compile(rf"\b(?:at|from|by|reach|reaching|reaches)\s+{_LEVEL}",
                           re.IGNORECASE)

# The strongest form: a sentence that STARTS with the grant, found anywhere in the text,
# even after flavour sentences ("...defend yourself from it. Beginning at level 3, you").
LEVEL_GRANT = re.compile(
    rf"(?:^|[.!?:]\s+|\n)(?:also\s+|additionally\s+)?(?:starting|beginning)?\s*"
    rf"(?:at|from|by|once you reach|when you reach)\s+{_LEVEL}",
    re.IGNORECASE)


def level_number(match: re.Match) -> int:
    """The level from a LEVEL_IN_TEXT / LEVEL_GRANT match, whichever word order it used."""
    return int(next(group for group in match.groups() if group))

# Level written in the heading itself: "Level 3: Frenzy", "3rd Level - Frenzy".
LEVEL_IN_HEADING = re.compile(
    rf"^\s*(?:level\s+(\d{{1,2}})|{_ORDINAL}[\s-]+level)\s*[:.\-–]\s*(?P<name>.+)$",
    re.IGNORECASE)

# Heading and level line merged by OCR into one block:
# "College of Puppetry Spells 3rd-level College of Puppetry feature".
HEADING_THEN_LEVEL_LINE = re.compile(
    rf"^(?P<name>.+?)\s+{_ORDINAL}[\s-]+level\b.{{0,70}}\bfeatures?\s*$", re.IGNORECASE)

# How far into a feature's text to look for its level. The grant phrase comes first;
# later sentences mention other levels ("cast it at 3rd level", "at 5th level or higher").
LEVEL_SEARCH_CHARS = 250

# =========================================================================================
# Blocks that aren't content
# =========================================================================================
# Credits, legal lines, tool signatures: "Homebrew by Wyll Hood", "Art by Olga Drebas",
# "Art Credit : Robson Michael", "Cover rart by Lizzart", "This is unofficial Fan Content
# permitted under the Fan Content Policy...", "©Wizards of the Coast LLC", "DD HOMEBREW".
# Only credit-shaped phrases: plain words like "made with" or "created by" also occur in
# rules ("checks made with disadvantage", "the creature created by your spell").
CREDITS = re.compile(
    r"\b(?:homebrew|art|artwork|cover\s+r?art|illustrations?|layout|writing|written|"
    r"design(?:ed)?)\s*(?:by|credits?)\b"
    r"|\b(?:homebrewery|gm ?binder|patreon|ko-?fi|artstation|deviantart)\b|https?://"
    r"|\b(?:unofficial fan content|fan content policy|wizards of the coast|approved/endorsed)\b"
    r"|©|^\W*d{0,2}\w?\s*homebrew\W*$"
    r"|^\W*(?:all\s+)?art\b[^.\n]{0,25}\bby\b",  # "All art created by me, u/Yorviing"
    re.IGNORECASE,
)
CREDITS_MAX_CHARS = 300  # longer blocks mentioning these are real content

# Headings whose whole section isn't subclass content: "Credits", "Version 1.0",
# "Special Thanks", "Changelog".
NON_CONTENT_HEADING = re.compile(
    r"^(?:credits?|special thanks|thanks|acknowledge?ments?|version\s*\d[\w.]*|v\d[\w.]*|"
    r"change\s*log|patch notes|about the author|legal)$", re.IGNORECASE)

# Headings of tables and lists: "Destructive Spell Table", "Shouts Table", "Minor Table",
# "Bard Level College Feature" (feature-summary table), "Roll Effect", "Rolled",
# "Fleshwarping Spell Tag List".
TABLE_HEADING = re.compile(
    r"\b(?:tables?|list)\b|^\w+\s+level\b.*\bfeatures?$|^(?:d\d+\s+)?roll(?:ed)?(?:\s+effects?)?$",
    re.IGNORECASE)

# Feature-summary tables (GM Binder template): header "Bard Level | College Feature", rows
# "3rd Commander's Training, Inspiring Orders 6th Gallant Leader 14th Masterful Commander".
# Read word by word: an ordinal token switches the level, other words are feature names
# (comma-separated). OCR may scramble the cells, so this is a hint, never the only source.
SUMMARY_TABLE_HEADER = re.compile(r"\blevel\b.*\bfeatures?\b", re.IGNORECASE)
ORDINAL_TOKEN = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)$", re.IGNORECASE)

# Spell-list sub-headings: "Cantrips", "2nd Level", "3rd-level spells".
SPELL_LIST_HEADING = re.compile(
    r"^(?:cantrips?(?:\s*\(0 level\))?|\d{1,2}(?:st|nd|rd|th)[\s-]+level(?:\s+spells)?)$",
    re.IGNORECASE)

# A spell description: every spell lists all of these.
SPELL_MARKERS = ("casting time", "range", "components", "duration")

# Group headers that introduce a list of subclasses or features but aren't one:
# "SubcLAss FEATures", "UPDATED CLASS FEATURES", "ROGUISH ARCHETYPES", "Arcane Traditions",
# "Optional Subclass Features", "14 CHAPTER 2 | SUBCLASSES", "A wizard subclass",
# "Fleshwarping Features" (a "<name> Features" header; real feature names are singular).
SECTION_HEADER = re.compile(
    r"^(?:(?:updated |optional )?(?:sub)?class features|[\w'’]+(?:\s+[\w'’]+)?\s+features|"
    r"(?:roguish |rogue |martial )?archetypes?|arcane traditions?|bard colleges?|"
    r"sacred oaths?|druid circles?|divine domains?|otherworldly patrons?|"
    r"sorcerous origins?|primal paths?|monastic traditions?|ranger conclaves?|"
    r"artificer specialists?|(?:a |an )?\w+ subclass(?:es)?|.*\bchapter \d+\b.*|subclasses)$",
    re.IGNORECASE,
)

# Stat blocks (summoned creatures, homunculi): both phrases appear in every stat block.
STAT_BLOCK_MARKERS = ("armor class", "hit points")

# =========================================================================================
# OCR and typesetting repairs (used by clean.py)
# =========================================================================================
# Letter O read instead of zero inside numbers: "1Oth level" -> "10th level".
OCR_O_IN_NUMBER = re.compile(r"(?<=\d)[Oo](?=\d|st\b|nd\b|rd\b|th\b)")
# "6oft" -> "60ft", "3oft" -> "30ft": o for 0 before a unit.
OCR_O_BEFORE_UNIT = re.compile(r"(?<=\d)o(?=\s*(?:ft|feet)\b)")
# OCR splits "14th" into "1 4th": a lone 1 followed by an ordinal. Real text never says
# "1 4th", so joining is safe.
OCR_SPLIT_ORDINAL = re.compile(r"\b1 (\d(?:st|nd|rd|th)\b)")
# Missing space after a sentence end: "tavern crowds.Where", "bravery,Fools".
MISSING_SPACE = re.compile(r"(?<=[a-z\)][.,;:!?])(?=[A-Z])")
# Inline bullets from OCR: "disadvantage.• You gain" -> new line per bullet.
INLINE_BULLET = re.compile(r"\s*[•●▪]\s*")
# Word broken across lines with a hyphen: "some-\nthing" -> "something".
HYPHEN_BREAK = re.compile(r"(?<=[a-z])-\n(?=[a-z])")
