"""PDF -> JSON. Modules in the order a PDF goes through them:

    convert.py      runs stages 1-5 for one PDF or a batch; errors, warnings, _report.json

    1  extract.py   PDF -> blocks CSV (data/blocks/): text layer or OCR. Standalone script,
                    because the OCR environment (.venv-ocr) runs it without this package
    2  clean.py     reading order, junk, drop caps, OCR repairs, level lines split off
    3a roles.py     each block's layout role: heading / body / table / level_line / other
    3b (classifier/block_classifier.py: body text -> description or rules)
    4  structure.py sections -> subclasses with a description, features and levels
    5  schema.py    the output format (pydantic): what the JSON contains, validated

    patterns.py     every wording rule (regexes) used by stages 2-4, with real examples
    check.py        scores data/output/ against hand-written data/expected/ (end to end)
"""
