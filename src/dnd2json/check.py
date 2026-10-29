"""End-to-end check: compare pipeline output with hand-written expected results.

evaluate.py tests one component (the classifier) on labelled blocks. This file tests the
whole pipeline the way a user sees it: does the JSON for a PDF have the right subclasses,
classes, descriptions, features and levels?

Expected files live in data/expected/<pdf name>.json and only list what matters, so
formatting differences don't count as errors:

    {"subclasses": [{
        "name": "College of Fools", "class": "Bard",
        "description_starts_with": "Bards who join the College of Fools",
        "features": [{"name": "Satirical Theatrics", "level": 3,
                      "description_starts_with": "You gain the skills to pass"}]}]}

Write them from the PDF, never from the pipeline's output: otherwise they repeat its
mistakes. Names are compared with structure.same_name (ignores casing/OCR noise).
"""

import difflib
import json
import re
from pathlib import Path

from dnd2json import config
from dnd2json.structure import same_name

# Texts "start the same" from this similarity on (0..1), to tolerate OCR typos.
START_MATCH_RATIO = 0.9


def _normal(text: str) -> str:
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def starts_with(actual: str, expected: str) -> bool:
    """Does `actual` begin with (roughly) `expected`? Empty expected = expect no text."""
    actual, expected = _normal(actual), _normal(expected)
    if not expected:
        return not actual
    head = actual[:len(expected)]
    similarity = difflib.SequenceMatcher(None, head, expected).ratio()
    return head == expected or similarity >= START_MATCH_RATIO


def compare_document(actual: dict, expected: dict) -> dict:
    """Score one document. Returns counts plus human-readable problems."""
    score = {"subclasses": 0, "subclasses_found": 0, "class_ok": 0, "description_ok": 0,
             "features_expected": 0, "features_actual": 0, "features_found": 0,
             "levels_ok": 0, "texts_ok": 0, "problems": []}
    problems = score["problems"]

    for exp_sub in expected["subclasses"]:
        score["subclasses"] += 1
        match = next((s for s in actual["subclasses"] if same_name(s["name"], exp_sub["name"])),
                     None)
        if match is None:
            problems.append(f"subclass missing: {exp_sub['name']}")
            score["features_expected"] += len(exp_sub["features"])
            continue
        score["subclasses_found"] += 1
        if match.get("class") == exp_sub.get("class"):
            score["class_ok"] += 1
        else:
            problems.append(f"{exp_sub['name']}: class {match.get('class')!r}, "
                            f"expected {exp_sub.get('class')!r}")
        if starts_with(match["description"], exp_sub.get("description_starts_with", "")):
            score["description_ok"] += 1
        else:
            problems.append(f"{exp_sub['name']}: description starts "
                            f"{match['description'][:60]!r}")

        score["features_expected"] += len(exp_sub["features"])
        score["features_actual"] += len(match["features"])
        for exp_feat in exp_sub["features"]:
            feat = next((f for f in match["features"] if same_name(f["name"], exp_feat["name"])),
                        None)
            if feat is None:
                problems.append(f"{exp_sub['name']}: feature missing: {exp_feat['name']}")
                continue
            score["features_found"] += 1
            if feat["level"] == exp_feat["level"]:
                score["levels_ok"] += 1
            else:
                problems.append(f"{exp_feat['name']}: level {feat['level']}, "
                                f"expected {exp_feat['level']}")
            if starts_with(feat["description"], exp_feat.get("description_starts_with", "")):
                score["texts_ok"] += 1
            else:
                problems.append(f"{exp_feat['name']}: text starts {feat['description'][:60]!r}")
        for feat in match["features"]:
            if not any(same_name(feat["name"], e["name"]) for e in exp_sub["features"]):
                problems.append(f"{exp_sub['name']}: unexpected feature: {feat['name']}")
    return score


def run_check(output_dir: Path = config.OUTPUT_DIR,
              expected_dir: Path = config.EXPECTED_DIR) -> dict:
    """Compare every expected file with its output. Prints a report, returns the totals."""
    expected_files = sorted(expected_dir.glob("*.json"))
    if not expected_files:
        print(f"No expected files in {expected_dir}")
        return {}

    totals: dict = {}
    for expected_path in expected_files:
        output_path = output_dir / expected_path.name
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        if not output_path.exists():
            print(f"[MISSING ] {expected_path.stem}: no output (run `dnd2json convert` first)")
            score = compare_document({"subclasses": []}, expected)
        else:
            actual = json.loads(output_path.read_text(encoding="utf-8"))
            score = compare_document(actual, expected)
            status = "OK" if not score["problems"] else "DIFF"
            print(f"[{status:8s}] {expected_path.stem}")
        for problem in score["problems"]:
            print(f"             - {problem}")
        for key, value in score.items():
            if key != "problems":
                totals[key] = totals.get(key, 0) + value

    def pct(part, whole):
        return f"{part}/{whole} ({100 * part / whole:.0f}%)" if whole else "n/a"

    print("\nTotals over", len(expected_files), "documents")
    print("  subclasses found   ", pct(totals["subclasses_found"], totals["subclasses"]))
    print("  class correct      ", pct(totals["class_ok"], totals["subclasses_found"]))
    print("  description correct", pct(totals["description_ok"], totals["subclasses_found"]))
    print("  feature recall     ", pct(totals["features_found"], totals["features_expected"]))
    print("  feature precision  ", pct(totals["features_found"], totals["features_actual"]))
    print("  level correct      ", pct(totals["levels_ok"], totals["features_found"]))
    print("  feature text start ", pct(totals["texts_ok"], totals["features_found"]))
    return totals
