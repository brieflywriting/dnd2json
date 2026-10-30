# Labels

One label per block. To label a block, go through the **rules in order**; the first rule
that fits decides. If a case isn't covered, decide once, add it to the borderline cases,
log it in the changelog, then label.

The labels describe **what the block will become in the JSON output**:
- `class_description` -> the subclass's intro text,
- `feature` -> one feature entry (name, level, rules text),
- `other` -> not part of the subclass entry (dropped or stored separately).

## The rules (apply in order)

1. **Two things in one block?** If a block holds parts with different labels (e.g. an
   intro followed by a feature heading and its text), it's a splitting error. **Split it**
   into separate blocks first, then label each part. Never label a merged block.

2. **Does it contain a game rule? -> `feature`.**
   A game rule changes what a character can do or how the game resolves something.
   Signs: a level ("3rd level"), dice ("2d6"), an action type (action, bonus action,
   reaction), a resource (spell slot, hit points, uses, charges, rests), a saving throw or
   DC, or "you gain / you can / you learn / you have advantage".
   Not a rule: mentioning game words as story ("they wield powerful spells", "you can roll
   on the table below for your patron's motivation").

3. **Does it introduce who/what a subclass is? -> `class_description`.**
   The identity, origin, reputation or philosophy of a subclass, or of a selectable option
   that works like one (a warlock patron, a pact, an oath). Second person ("You are...",
   "Your magic draws...") counts too: the pronoun doesn't decide the label, the content does.

4. **Anything else -> `other`.**
   Titles and headings on their own, tables, sidebars, credits, art captions, page
   headers/footers, and intros to a whole book or list rather than to one subclass.

## `class_description`

**Definition:** the introduction of one subclass (or subclass-like option): who its
members are and what drives them, with no game rules.

**Examples:**
- "Bards of the College of the Voice ridicule paltry singers and tellers of tawdry tales…" (Bard - College of the Voice)
- "Paladins that take the Oath of Martyrdom lay down their bodies for the safety and wellbeing of others…" (Paladin - Oath of Martyrdom)

## `feature`

**Definition:** a block that contains at least one game rule granted by the subclass,
including any flavor sentences around that rule.

**Examples:**
- "14th-level College of Drama feature. The Muse grants you the ability to enthrall Death itself… Your Masks grant the benefits below…" (Bard - College of Drama) — has a level.
- "Your bardic magic is infused with the power of the lower planes. When a creature under the effect of Countercharm succeeds on a saving throw…" (Bard - College of Crossroads) — no level, but a saving throw rule.

## `other`

**Definition:** any block that isn't a subclass intro or a feature.

**Examples:** none labelled yet. The current CSVs only contain the two other labels; add
`other` when you label full documents.

## Decision rules for borderline cases

1. **Rules text that opens a document.**
   "Your patron marks your body with a symbol… Whenever you expend a warlock spell slot,
   your mark gains a charge…" (Pact Boon - Pact of the Brand, NewClasses row 55)
   → **`feature`** (rule 2).
   Why: it contains rules (charges, spell slots, long rest). Position in the document
   doesn't decide the label, content does.

2. **A description written in second person.**
   "You are an Influencer, a duplicitous manipulator…" (Rogue - Influencer)
   "You have made a pact with a being that you share a significant fondness with…" (Warlock - Beloved)
   → **`class_description`** (rule 3).
   Why: no game rule; it introduces who the subclass is. "You can roll for a possible
   reason in the table" is story, not a rule.

3. **An intro to a whole book or option list, not to one subclass.**
   "For those with the willpower and time, arcane magic can be a path to great power…"
   (Patrons, Boons, and Invocations, NewClasses row 42)
   → **`other`** (rule 4).
   Why: it introduces a collection, not one subclass. In the JSON it belongs to no subclass.

4. **Flavor text that introduces a patron/boon/option inside a bigger document.**
   "Grandmother, auntie, wood witch, medicine woman, crone; the strange creatures known as
   hags…" (Patrons, Boons, and Invocations, NewClasses row 45)
   → **`class_description`** (rule 3).
   Why: a warlock patron is a subclass; this is its intro.

5. **A block that holds two things.**
   "The Chocolatier is a whimsical wizard… Cooking Prodigy / 2nd-level Chocolatier Magic
   feature / You are an academic…" (Wizard - Chocolatier, OCR row 24)
   → **split** (rule 1): the intro is `class_description`, "Cooking Prodigy…" onwards is `feature`.

6. **Flavor sentence(s) at the start of a feature.**
   "Your bardic magic is infused with the power of the lower planes. When a creature…"
   → **`feature`** (rule 2).
   Why: the flavor is part of the feature's text; the block contains a rule.

7. **A description that mentions mechanics in passing.**
   "…they gain access to powerful spells", "…a master of the bonus action"
   → **`class_description`** (rule 3), as long as no actual rule is stated.
   Test: could a player use the sentence at the table to resolve something? If not, it's
   not a rule.

8. **An epigraph or quote before the intro.**
   "'Hope is a powerful thing, it has carried many people out of darkness…'" followed by
   the intro (Paladin - Oath of Hope)
   → same block as the intro: **`class_description`**. On its own (separate block): `other`.

## Changelog

Record every rule change with the date, and which rows you relabelled because of it.

- 2026-10-29: the three relabels below applied. `other` is now an allowed label
  (tests/test_data.py). The classifier ignores `other` rows
  (classifier/block_classifier.py): page furniture is recognised by rules in
  pipeline/roles.py, not by the model.
  Also fixed the 14 rows with `�` in `5e-OCR-CombinedNewClasses.csv` (apostrophes, one
  dash, quotes, bullets). Row 27 still has `???` where OCR lost words: check the PDF.
- 2026-10-28: rules written. Relabels these decisions imply:
  - `5e-OCR-CombinedNewClasses.csv` row 55 (Pact of the Brand): `class_description` -> `feature`.
  - `5e-OCR-CombinedNewClasses.csv` row 42 (Patrons, Boons, and Invocations intro):
    `class_description` -> `other`. Needs `other` added to the allowed labels first
    (tests/test_data.py `LABELS`, classifier/evaluate.py `LABELS`), or drop the row until then.
  - `5e-OCR-Combined.csv` row 24 (Chocolatier): split into two rows.
