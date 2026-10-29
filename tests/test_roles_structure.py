"""Stages 3a (roles.py) and 4 (structure.py) on hand-made blocks: no PDFs, no model.

Blocks are given a `p_desc` directly (the classifier's job), so these tests check the
structure logic alone.
"""

import pandas as pd

from dnd2json import roles, structure


def row(text, role=None, size=9.0, source="text", label="", p_desc=0.1, bold=0, page=1):
    return dict(text=text, role=role, font_size=size, source=source, layout_label=label,
                p_desc=p_desc, is_bold=bold, page=page)


# --- roles -------------------------------------------------------------------------------
def test_roles_text_layer():
    body = 9.0
    assert roles.classify_role(row("Chaotic Rift", size=15), body)[0] == "heading"
    assert roles.classify_role(row("Arcane Vault", size=9, bold=1), body)[0] == "heading"
    assert roles.classify_role(row("Armor Class 13", size=10, bold=1), body)[0] == "table"
    assert roles.classify_role(row("3rd-level College of Fools feature"), body)[0] == "level_line"
    assert roles.classify_role(row("Homebrew by Wyll Hood", size=12), body)[0] == "other"
    assert roles.classify_role(row("At 3rd level, your magic can strike."), body)[0] == "body"


def test_roles_ocr_level_line_beats_layout_label():
    r = row("6th-level College of Revelry feature", source="ocr", label="vision_footnote")
    assert roles.classify_role(r, 10.0)[0] == "level_line"


def test_roles_ocr_long_image_text_is_body():
    r = row("Your mastery of music allows you to soothe any creature. " * 4, source="ocr",
            label="image")
    assert roles.classify_role(r, 10.0)[0] == "body"


# --- names ---------------------------------------------------------------------------------
def test_parse_filename():
    assert structure.parse_filename("Bard - College of Chaos_2") == ("Bard", "College of Chaos")
    assert structure.parse_filename("acolelgeofflavour") == (None, None)


def test_fix_case():
    assert structure.fix_case("COLLEGe Of CHaNCE") == "College of Chance"
    assert structure.fix_case("Extra ATtack") == "Extra Attack"
    assert structure.fix_case("Embrace the Chaos") == "Embrace the Chaos"  # untouched


def test_same_name_ignores_prefixes_and_ocr_noise():
    assert structure.same_name("Roguish Archetype: Mad Dog", "Mad Dog")
    assert structure.same_name("SCHoOLOF MIMIC MAGIC", "School of Mimic Magic")
    assert structure.same_name("WIZARD COMmUNITY COLLEGE", "Community College")
    assert not structure.same_name("College of Sirens Spells", "Siren's Song")


def test_split_heading_level():
    assert structure.split_heading_level("Level 3: Frenzy") == (3, "Frenzy")
    assert structure.split_heading_level(
        "College of Puppetry Spells 3rd-level College of Puppetry feature") == (
        3, "College of Puppetry Spells")
    assert structure.split_heading_level("Frenzy") == (None, "Frenzy")


def test_parse_summary_table():
    text = ("Bard Level College Feature 3rd Charming Presence, Depraved Enchantments "
            "6th Stirring Performance 14th Hedonistic Trance")
    assert structure.parse_summary_table(text) == {
        "Charming Presence": 3, "Depraved Enchantments": 3,
        "Stirring Performance": 6, "Hedonistic Trance": 14}
    assert structure.parse_summary_table("d4 1st Level Random Spell") == {}  # not a summary


# --- whole documents -----------------------------------------------------------------------
def build(blocks, stem="Bard - College of Tests"):
    df = pd.DataFrame(blocks)
    return structure.build_subclasses(df, stem)


def test_level_line_document_with_sub_options():
    subclasses, _ = build([
        row("College of Tests", "heading"),
        row("Bards of this college test everything.", "body", p_desc=0.9),
        row("Channel the Muse", "heading"),
        row("3rd-level College of Tests feature", "level_line"),
        row("You manifest a mask, granting the abilities below:", "body"),
        row("Mask of Comedy", "heading"),
        row("You inspire your allies.", "body"),
        row("Soliloquy", "heading"),
        row("6th-level College of Tests feature", "level_line"),
        row("Your song is infused with power.", "body"),
    ])
    (sub,) = subclasses
    assert sub.class_name == "Bard"
    assert sub.description == "Bards of this college test everything."
    assert [(f.name, f.level) for f in sub.features] == [("Channel the Muse", 3), ("Soliloquy", 6)]
    assert "Mask of Comedy. You inspire your allies." in sub.features[0].description


def test_stat_block_and_table_and_credits_section():
    subclasses, warnings = build([
        row("Homunculus", "heading"),
        row("At 2nd level, you make a homunculus.", "body"),
        row("Destructive Spell Table", "heading"),
        row("1 Burning Hands Fireball", "table"),
        row("Cocoa Homunculus", "heading"),
        row("Armor Class 13 Hit Points 5", "table"),
        row("Actions", "heading"),
        row("Chocolate Blob. Ranged Weapon Attack", "table"),
        row("Credits", "heading"),
        row("Thanks to everyone.", "body", p_desc=0.9),
    ], stem="Wizard - Chocolatier")
    (sub,) = subclasses
    assert [f.name for f in sub.features] == ["Homunculus"]
    assert sub.tables[0].title == "Destructive Spell Table"
    (stat,) = [e for e in sub.extra_sections if e.kind == "stat_block"]
    assert "Actions" in stat.text  # the follow-up section joined the stat block
    assert not any("Thanks" in e.text for e in sub.extra_sections)
    assert any("no description" in w for w in warnings)


def test_multi_subclass_file_is_split():
    subclasses, _ = build([
        row("Arcane Traditions intro text for the whole book.", "body", p_desc=0.9),
        row("School of Reawakening", "heading"),
        row("You combine necromancy and conjuration.", "body", p_desc=0.9),
        row("Dual Savant", "heading"),
        row("At 2nd level, copying is halved.", "body"),
        row("School of Hemomancy", "heading"),
        row("Blood magic, studied.", "body", p_desc=0.9),
        row("Deepen Wound", "heading"),
        row("Starting at 6th level, wounds deepen.", "body"),
    ], stem="Wizard - Schools of Reawakening & Hemomancy")
    assert [s.name for s in subclasses] == ["School of Reawakening", "School of Hemomancy"]
    assert [f.name for f in subclasses[1].features] == ["Deepen Wound"]
    assert subclasses[0].extra_sections[0].kind == "preamble"


def test_features_sorted_by_level():
    # OCR put the 17th-level feature first. A feature without a level stays right after
    # the feature it followed in the document ("Early"), wherever that one moves.
    features = [structure.Feature(name="Late", level=17, description=""),
                structure.Feature(name="Early", level=3, description=""),
                structure.Feature(name="Unknown", level=None, description="")]
    assert [f.name for f in structure.order_features(features)] == ["Early", "Unknown", "Late"]
