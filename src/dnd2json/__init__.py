"""dnd2json: D&D 5e homebrew subclass PDFs -> structured JSON.

Package map (each module's docstring explains it in detail):

    cli.py          the `dnd2json` command: train, convert, check, extract, evaluate
    config.py       every path and file name, in one place
    errors.py       exception types (what can go wrong, and where)

    pipeline/       the product: PDF -> JSON, one module per stage
    classifier/     the machine-learning part: is a paragraph a description or rules?

The two subpackages meet in one place: pipeline/convert.py calls the trained classifier
from classifier/block_classifier.py (stage 3b).

The __init__.py files only hold documentation. Import modules by their full path
(`from dnd2json.pipeline.convert import convert_many`), so it's always visible where a
function lives, and the subpackages can import each other without import cycles.
"""
