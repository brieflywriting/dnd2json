"""Stage 4: turn role-labelled blocks into subclasses with a description and features.

The document is read top to bottom as a sequence of SECTIONS: a heading plus the blocks
under it, up to the next heading. Then every section is decided:

    intro (blocks under the title, or before any heading)
        -> description, block by block (the classifier filters out non-intro text)
    "Credits", "Version 1.0"                                -> skipped
    contains a stat block ("Armor Class" + "Hit Points")   -> extra section, kind stat_block
    only tables, or a table/list heading ("Shouts Table")  -> table
    a spell ("Casting Time", "Range", "Components"...)    -> extra section, kind spell
    has a level (level line, "Level 3:" heading, "At 3rd level" in the text) -> feature
    no level, description-like text  -> description (if none yet) or lore
    no level, rules-like text        -> part of the feature above (a sub-option such as
                                        "Mask of Comedy"), or a feature without a level

Most PDFs hold one subclass. A file named like "Schools of Reawakening & Hemomancy" holds
several: the document is then split at each subclass title first.

Every function works on plain data (lists of dicts and small dataclasses) so it can be
tested without PDFs: see tests/test_roles_structure.py.
"""

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from dnd2json.pipeline import patterns
from dnd2json.pipeline.schema import ExtraSection, Feature, Subclass, Table

# P(class_description) at or above this counts as description-like.
DESCRIPTION_THRESHOLD = 0.5
# Two subclass names are "the same" from this similarity on (letters only, 0..1).
NAME_MATCH_RATIO = 0.85
# Words that never distinguish subclass names ("Roguish Archetype: Mad Dog" = "Mad Dog").
NAME_NOISE_WORDS = {c.lower() for c in patterns.CLASS_NAMES} | {
    "roguish", "archetype", "arcane", "tradition", "martial", "subclass", "homebrew"}
# Fewer level lines than this: the document doesn't use the level-line template.
LEVEL_LINE_TEMPLATE_MIN = 2
# Small words kept lower-case when fixing the casing of a heading.
SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the",
               "to", "with", "from"}
# Filenames naming several subclasses: "Schools of Reawakening & Hemomancy".
MULTI_SUBCLASS_FILENAME = re.compile(r"&|\band\b|^(?:schools|colleges|oaths|circles|paths)\b",
                                     re.IGNORECASE)


# =========================================================================================
# Names
# =========================================================================================
def parse_filename(stem: str) -> tuple[str | None, str | None]:
    """'Bard - College of Chaos_2' -> ('Bard', 'College of Chaos'). (None, None) if the
    name doesn't follow the 'Class - Subclass' pattern."""
    match = patterns.FILENAME.match(stem.strip())
    if not match:
        return None, None
    cls = match["cls"].capitalize()
    return (cls if cls in patterns.CLASS_NAMES else None), match["name"].strip()


def fix_case(text: str) -> str:
    """Repair OCR casing: 'COLLEGe Of CHaNCE' -> 'College of Chance'.

    Only applied when the casing looks broken (mostly capitals, or a capital after a
    lower-case letter inside a word like 'FooL'); normal headings are left untouched.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return text
    upper_ratio = sum(c.isupper() for c in letters) / len(letters)
    # "FooL" (capital after lower-case) or "ATtack" (two capitals, then lower-case).
    broken = re.search(r"[a-z][A-Z]|\b[A-Z]{2,}[a-z]", text) is not None
    if upper_ratio < 0.5 and not broken:
        return text
    words = text.lower().split()
    fixed = [w if (i > 0 and w in SMALL_WORDS) else w[:1].upper() + w[1:]
             for i, w in enumerate(words)]
    return " ".join(fixed)


def clean_heading(text: str) -> str:
    """Heading text -> display name: fixed casing, no trailing punctuation."""
    text = " ".join(text.split())
    return fix_case(text).strip(" .:;-–")


def strip_title_prefix(title: str) -> str:
    """'Bard: College of Chivalry' -> 'College of Chivalry';
    "Giruza's School of Mythos" -> 'School of Mythos'."""
    title = patterns.TITLE_PREFIX.sub("", title)
    title = re.sub(r"^[\w’']+['’]s\s+(?=\w)", "", title)  # author's possessive
    return title.strip()


def name_key(text: str) -> str:
    """Comparison key: lower-case letters only, without class names and template words."""
    words = [w for w in re.findall(r"[a-z]+", text.lower()) if w not in NAME_NOISE_WORDS]
    return "".join(words)


def same_name(a: str, b: str) -> bool:
    ka, kb = name_key(a), name_key(b)
    if not ka or not kb:
        return False
    return ka == kb or difflib.SequenceMatcher(None, ka, kb).ratio() >= NAME_MATCH_RATIO


def class_label(text: str) -> str | None:
    """A heading that is only a class name ('WIZARD', 'Bard') -> that class."""
    stripped = text.strip().capitalize()
    return stripped if stripped in patterns.CLASS_NAMES else None


def class_from_subclass_name(name: str) -> str | None:
    """'College of Chaos' -> 'Bard', via the prefix table in patterns.py."""
    lowered = strip_title_prefix(name).lower()
    for prefix, cls in patterns.SUBCLASS_PREFIX_TO_CLASS.items():
        if lowered.startswith(prefix):
            return cls
    return None


def class_from_text(texts) -> str | None:
    """Most mentioned class name in the document, as a last resort."""
    counts = Counter()
    for text in texts:
        for cls in patterns.CLASS_NAMES:
            counts[cls] += len(re.findall(rf"\b{cls}s?\b", text, re.IGNORECASE))
    best = counts.most_common(1)
    return best[0][0] if best and best[0][1] > 0 else None


# =========================================================================================
# Levels
# =========================================================================================
def level_from_line(text: str) -> int | None:
    match = patterns.LEVEL_LINE.match(text)
    return int(match.group(1)) if match else None


def level_from_text(text: str) -> int | None:
    """Level from a feature's text.

    First choice: a sentence that starts with the grant ("Beginning at 3rd level, ..."),
    anywhere in the text, since flavour sentences may come first. Fallback: any grant
    phrase near the start ("When you choose this archetype at 3rd level").
    """
    match = patterns.LEVEL_GRANT.search(text)
    if match is None:
        match = patterns.LEVEL_IN_TEXT.search(text[:patterns.LEVEL_SEARCH_CHARS])
    return patterns.level_number(match) if match else None


def split_heading_level(heading: str) -> tuple[int | None, str]:
    """Pull a level out of a heading, returning (level, name).

    'Level 3: Frenzy' -> (3, 'Frenzy')
    'College of Puppetry Spells 3rd-level College of Puppetry feature'
        -> (3, 'College of Puppetry Spells')     (OCR merged heading and level line)
    'Frenzy' -> (None, 'Frenzy')
    """
    match = patterns.LEVEL_IN_HEADING.match(heading)
    if match:
        return int(match.group(1) or match.group(2)), match["name"].strip()
    match = patterns.HEADING_THEN_LEVEL_LINE.match(heading)
    if match:
        return int(match.group(2)), match["name"].strip()
    return None, heading


# =========================================================================================
# Feature-summary tables
# =========================================================================================
SUMMARY_MIN_KEY_CHARS = 5  # shorter names are too generic to match by containment
SUMMARY_ROW_MAX_CHARS = 200  # a summary row lists names; longer blocks are paragraphs
SUMMARY_TRUST_MIN_ENTRIES = 3  # names that must match real headings to trust the table


def parse_summary_table(text: str) -> dict[str, int]:
    """'Bard Level College Feature 3rd A, B 6th C' -> {'A': 3, 'B': 3, 'C': 6}.

    Returns {} for tables that aren't feature summaries (no "Level ... Feature" header).
    """
    if not patterns.SUMMARY_TABLE_HEADER.search(text):
        return {}
    entries, level, words = {}, None, []

    def flush():
        for name in re.split(r",|\band\b", " ".join(words)):
            name = clean_heading(name)
            if level is not None and name and not patterns.SUMMARY_TABLE_HEADER.search(name):
                entries[name] = level

    for token in text.split():
        ordinal = patterns.ORDINAL_TOKEN.match(token)
        if ordinal:
            flush()
            level, words = int(ordinal.group(1)), []
        else:
            words.append(token)
    flush()
    return entries


def summary_matches(name: str, entry: str) -> bool:
    """Does a heading match a summary-table entry? Same name, or one contains the other
    (OCR may merge two cells: 'Thespian's Craft Soliloquy')."""
    if same_name(name, entry):
        return True
    a, b = name_key(name), name_key(entry)
    long_enough = min(len(a), len(b)) >= SUMMARY_MIN_KEY_CHARS
    return long_enough and (a in b or b in a)


def summary_level(name: str, summary: dict[str, int]) -> int | None:
    for entry, level in summary.items():
        if summary_matches(name, entry):
            return level
    return None


# =========================================================================================
# Sections
# =========================================================================================
@dataclass
class Section:
    heading: str | None             # cleaned heading text; None for text before any heading
    is_title: bool = False          # heading is the subclass title (its blocks are the intro)
    level: int | None = None
    blocks: list[dict] = field(default_factory=list)  # body and table blocks, in order

    @property
    def body(self) -> list[dict]:
        return [b for b in self.blocks if b["role"] == "body"]

    @property
    def tables(self) -> list[dict]:
        return [b for b in self.blocks if b["role"] == "table"]

    def text(self, role: str = "body") -> str:
        return "\n\n".join(b["text"] for b in self.blocks if b["role"] == role)

    def description_probability(self) -> float:
        """Length-weighted mean P(description) of the body blocks."""
        body = self.body
        total = sum(len(b["text"]) for b in body)
        if not total:
            return 0.0
        return sum(b["p_desc"] * len(b["text"]) for b in body) / total


def build_sections(blocks: list[dict], is_title) -> tuple[list[Section], list[str]]:
    """Group blocks into sections. Returns the sections and class names seen as labels."""
    sections: list[Section] = []
    class_labels: list[str] = []
    current: Section | None = None

    for block in blocks:
        role = block["role"]
        if role == "other":
            continue
        if role == "heading":
            cls = class_label(block["text"])
            if cls:                       # "WIZARD" above the title: a hint, not a section
                class_labels.append(cls)
                continue
            level, name = split_heading_level(clean_heading(block["text"]))
            current = Section(heading=name, is_title=is_title(block["text"]), level=level)
            sections.append(current)
        elif role == "level_line":
            if current is None:
                current = Section(heading=None)
                sections.append(current)
            if current.level is None:
                current.level = level_from_line(block["text"])
        else:  # body or table
            if current is None:
                current = Section(heading=None)
                sections.append(current)
            current.blocks.append(block)
    return sections, class_labels


# =========================================================================================
# Deciding what each section is
# =========================================================================================
def _join(*parts: str) -> str:
    return "\n\n".join(p for p in parts if p)


def decide_sections(sections: list[Section], uses_level_lines: bool,
                    summary: dict[str, int] | None = None):
    """Turn sections into (description, features, extras, tables).

    `summary` maps feature names to levels from the document's feature-summary table, if
    it has one ({} otherwise). Names in it are always features, never sub-options.
    """
    summary = summary or {}
    # Trust the summary only if it's confirmed by the document: enough of its names are
    # real headings. A spell table can look like a summary ("3rd Echolocate, ...") and
    # must not turn the real features into sub-options.
    headings = [s.heading for s in sections if s.heading]
    confirmed = sum(any(summary_matches(h, entry) for h in headings) for entry in summary)
    trust_summary = confirmed >= SUMMARY_TRUST_MIN_ENTRIES
    description_parts: list[str] = []
    features: list[Feature] = []
    extras: list[ExtraSection] = []
    tables: list[Table] = []
    stat_block: ExtraSection | None = None  # open stat block: following tables join it

    for section in sections:
        body_text, table_text = section.text("body"), section.text("table")
        name = section.heading or ""
        if not body_text and not table_text:
            continue  # heading without content: art caption, group header
        if patterns.NON_CONTENT_HEADING.match(name):
            continue  # "Credits", "Version 1.0": not subclass content

        # --- stat blocks (creature statistics) ----------------------------------------------
        # A section with a level line is a feature even if stat-block text ended up inside
        # it (OCR sometimes files a sidebar stat block under the feature next to it).
        all_text = _join(body_text, table_text).lower()
        if section.level is None and all(m in all_text for m in patterns.STAT_BLOCK_MARKERS):
            stat_block = ExtraSection(name=name or "Stat block", kind="stat_block",
                                      text=_join(body_text, table_text))
            extras.append(stat_block)
            continue
        if stat_block is not None and not body_text:
            # "Actions" / "Reactions" right after a stat block belong to it.
            addition = f"{name}\n{table_text}" if name else table_text
            stat_block.text = _join(stat_block.text, addition)
            continue
        stat_block = None

        # --- the intro: everything under the title or before the first heading --------------
        if section.is_title or section.heading is None:
            preamble = []
            for block in section.body:
                if block["p_desc"] >= DESCRIPTION_THRESHOLD:
                    description_parts.append(block["text"])
                else:
                    preamble.append(block["text"])  # e.g. "At 3rd level, a bard gains..."
            if preamble:
                extras.append(ExtraSection(name="Preamble", kind="preamble",
                                           text=_join(*preamble)))
            if table_text:
                tables.append(Table(title=None, text=table_text))
            continue

        # --- tables and lists --------------------------------------------------------------
        # A table heading ("Shouts Table", "Bard Level College Feature") or a spell-list
        # heading ("2nd Level") makes its whole section a table, even rows set as body.
        if (not body_text or patterns.TABLE_HEADING.search(name)
                or patterns.SPELL_LIST_HEADING.match(name)):
            tables.append(Table(title=name or None, text=_join(body_text, table_text)))
            continue

        # --- spell descriptions ("Casting Time: 1 action, Range: ...") ----------------------
        if all(marker in all_text for marker in patterns.SPELL_MARKERS):
            extras.append(ExtraSection(name=name, kind="spell", text=_join(body_text, table_text)))
            continue

        # Level sources, most reliable first: level line / heading, the rules text, then
        # the summary table (OCR can scramble its cells).
        in_summary = summary_level(name, summary) if name else None
        level = section.level or level_from_text(body_text) or in_summary
        p_desc = section.description_probability()
        # With a trustworthy summary table, a rules section it doesn't list is a sub-option
        # ("Speed of the Deer" under "Primal Inspiration"), even if OCR misplaced a level
        # line under it.
        unlisted_option = (trust_summary and in_summary is None
                           and bool(features) and p_desc < DESCRIPTION_THRESHOLD)

        # --- features ------------------------------------------------------------------
        if unlisted_option:
            features[-1].description = _join(features[-1].description, f"{name}. {body_text}")
        elif level is not None:
            features.append(Feature(name=name, level=level, description=body_text))
        elif p_desc >= DESCRIPTION_THRESHOLD:
            if not description_parts and not features:
                # The intro sits under a subtitle ("An Arcane Tradition") instead of
                # directly under the title: it's still the description.
                description_parts.append(body_text)
            else:
                # Lore or a sidebar ("Roleplaying a Fool"): kept apart, with its heading.
                extras.append(ExtraSection(name=name, kind="lore", text=body_text))
        elif features and (uses_level_lines or features[-1].description.rstrip().endswith(":")):
            # A sub-option of the feature above ("Mask of Comedy" under "Channel the Muse"):
            # the document announces every feature's level, and this heading has none.
            features[-1].description = _join(features[-1].description, f"{name}. {body_text}")
        else:
            features.append(Feature(name=name, level=None, description=body_text))

        if table_text:  # tables inside a feature (e.g. a feature-summary table)
            tables.append(Table(title=None, text=table_text))

    return _join(*description_parts), features, extras, tables


def order_features(features: list[Feature]) -> list[Feature]:
    """Sort by level, keeping the document order for equal levels.

    OCR sometimes returns a page in the wrong order (a 17th-level feature first).
    Features without a level keep their place: they sort as the level before them.
    """
    keyed, last = [], 0
    for position, feature in enumerate(features):
        if feature.level is not None:
            last = feature.level
        keyed.append((last, position, feature))
    return [feature for _, _, feature in sorted(keyed, key=lambda k: (k[0], k[1]))]


# =========================================================================================
# Whole document
# =========================================================================================
def find_titles(blocks: list[dict], multi: bool) -> list[int]:
    """Positions (indexes into `blocks`) of headings that start a new subclass.

    One-subclass files: none (the document isn't split; title headings only mark the
    intro). Multi-subclass files: every heading that looks like a subclass name
    ("School of Hemomancy"), first occurrence of each name only.
    """
    if not multi:
        return []
    starts, seen = [], []
    for i, block in enumerate(blocks):
        if block["role"] != "heading":
            continue
        title = strip_title_prefix(clean_heading(block["text"]))
        if patterns.SUBCLASS_TITLE.match(title) and not any(same_name(title, s) for s in seen):
            seen.append(title)
            starts.append(i)
    return starts if len(starts) >= 2 else []


def build_subclasses(df: pd.DataFrame, source_stem: str) -> tuple[list[Subclass], list[str]]:
    """The stage entry point: blocks (with role and p_desc columns) -> subclasses."""
    file_class, file_name = parse_filename(source_stem)
    multi = bool(file_name and MULTI_SUBCLASS_FILENAME.search(file_name))
    blocks = df.to_dict("records")
    warnings: list[str] = []

    title_positions = find_titles(blocks, multi)
    preamble_blocks: list[dict] = []
    if title_positions:
        # Text before the first title is the book's general intro ("The study of wizardry
        # is ancient..."), not any one subclass's description: kept as a preamble.
        preamble_blocks = blocks[:title_positions[0]]
        bounds = title_positions + [len(blocks)]
        segments = [blocks[a:b] for a, b in zip(bounds, bounds[1:], strict=False)]
        segment_titles = [strip_title_prefix(clean_heading(blocks[p]["text"]))
                          for p in title_positions]
        warnings.append(f"file holds {len(segments)} subclasses: {', '.join(segment_titles)}")
    else:
        segments, segment_titles = [blocks], [None]

    uses_level_lines = sum(b["role"] == "level_line" for b in blocks) >= LEVEL_LINE_TEMPLATE_MIN
    all_body_text = [b["text"] for b in blocks if b["role"] == "body"]

    subclasses = []
    for segment, segment_title in zip(segments, segment_titles, strict=True):
        # The subclass name: the file name when the file holds one subclass (cleanest),
        # else the title heading; for odd file names, the first title-like heading.
        name = segment_title or file_name or _first_title_heading(segment) or source_stem

        def is_title(text, name=name):
            return same_name(strip_title_prefix(clean_heading(text)), name)

        sections, class_labels = build_sections(segment, is_title)
        summary = find_summary(segment)
        description, features, extras, tables = decide_sections(
            sections, uses_level_lines, summary)
        warnings += _summary_warnings(name, summary, features)

        preamble = _join(*(b["text"] for b in preamble_blocks if b["role"] == "body"))
        if preamble and not subclasses:  # attach the book intro to the first subclass only
            extras.insert(0, ExtraSection(name="Preamble", kind="preamble", text=preamble))

        cls = (file_class or (class_labels[0] if class_labels else None)
               or class_from_subclass_name(name) or class_from_text(all_body_text))
        subclass = Subclass(name=name, description=description,
                            features=order_features(features), extra_sections=extras,
                            tables=tables, **{"class": cls})
        warnings += _subclass_warnings(subclass)
        subclasses.append(subclass)
    return subclasses, warnings


def find_summary(blocks: list[dict]) -> dict[str, int]:
    """The feature-summary table of a document, as {feature name: level}.

    Its header ("Bard Level College Feature") and rows ("3rd Charming Presence, ...")
    are often separate blocks, even in different sections when OCR reorders them, so the
    header blocks and the row-like table blocks of the whole segment are read together.
    """
    parts = []
    for block in blocks:
        text = block["text"]
        is_header = (block["role"] in ("heading", "table")
                     and patterns.SUMMARY_TABLE_HEADER.search(text))
        # Rows: short blocks starting with a bare ordinal ("3rd Charming Presence, ...").
        # OCR files them as table or as body text, so both count.
        first_word = text.split()[0] if text.split() else ""
        is_row = (block["role"] in ("table", "body") and len(text) <= SUMMARY_ROW_MAX_CHARS
                  and patterns.ORDINAL_TOKEN.match(first_word))
        if is_header or is_row:
            parts.append(text)
    return parse_summary_table(" ".join(parts))


def _summary_warnings(name: str, summary: dict[str, int], features: list[Feature]) -> list[str]:
    """Features the summary table lists but the document didn't yield (lost to OCR).

    Only for confirmed summaries (most entries found), so a spell table that merely looks
    like a summary doesn't produce a list of "missing features".
    """
    missing = [entry for entry in summary
               if not any(summary_matches(f.name, entry) for f in features)]
    found = len(summary) - len(missing)
    if not missing or found < SUMMARY_TRUST_MIN_ENTRIES:
        return []
    return [f"{name}: summary table lists features not found in the text: {', '.join(missing)}"]


def _first_title_heading(blocks: list[dict]) -> str | None:
    for block in blocks:
        if block["role"] == "heading":
            title = strip_title_prefix(clean_heading(block["text"]))
            if patterns.SUBCLASS_TITLE.match(title):
                return title
    return None


def _subclass_warnings(subclass: Subclass) -> list[str]:
    prefix = f"{subclass.name}: "
    warnings = []
    if not subclass.description:
        warnings.append(prefix + "no description found")
    if not subclass.features:
        warnings.append(prefix + "no features found")
    no_level = [f.name for f in subclass.features if f.level is None]
    if no_level:
        warnings.append(prefix + f"no level found for {', '.join(no_level)}")
    if subclass.class_name is None:
        warnings.append(prefix + "class not found")
    return warnings
