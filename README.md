# dnd2json

###Currently a WIP. The pipeline requires testing and changes.

Converts D&D 5e homebrew subclass PDFs into structured JSON: subclass name, class,
description and features (name, level, rules text), plus tables, stat blocks and lore.

```
dnd2json train                  # once: train the block classifier on data/gold
dnd2json convert --all          # every PDF in media/eng/pdf -> data/output/*.json
dnd2json check                  # score the output against data/expected
```

(`dnd2json` is installed by `pip install -e ".[dev]"`; `python -m dnd2json.cli` works too.)

| Document | What's in it |
|---|---|
| `README.md` | how to run, folders, how to debug (this file) |
| `PLAN.md` | steps done and to do, with checks |
| `NEXT.md` | the one step to do next |
| `DECISIONS.md` | why things are built this way |
| `LABELS.md` | labelling rules for the gold data |

## The pipeline

```
PDF ─► 1 extract ─► 2 clean ─► 3 roles + classifier ─► 4 structure ─► 5 JSON
       blocks CSV    order,       heading/body/table/     sections ->     validated
       per PDF       junk,        level_line/other,       subclasses,     (schema.py)
                     drop caps    P(description)          features
```

| Stage | File | What it does |
|---|---|---|
| 1 | `extract.py` | PDF -> one row per text block. Text-layer PDFs via PyMuPDF (font size, bold, italic); scanned or garbled PDFs via PaddleOCR (layout labels, confidence). Cached in `data/blocks/`. |
| 2 | `clean.py` | Column reading order, junk removal, drop caps ("C"+"haotic"), OCR digit fixes ("1Oth"), line unwrapping, level lines split off paragraphs. |
| 3a | `roles.py` | Each block's layout role, with the rule that decided it (`role_reason`). |
| 3b | `classifier.py` | TF-IDF + logistic regression (`words_only` from `models.py`): is body text a description or rules? |
| 4 | `structure.py` | Groups blocks into sections under headings and decides what each one is: description, feature (+ level), sub-option, table, stat block, spell, lore. Splits files holding several subclasses. |
| 5 | `schema.py` | The output format (pydantic): documents what the JSON contains and validates it. |

Supporting files: `pipeline.py` (runs the stages, error handling, batch), `patterns.py`
(all wording rules, with real examples), `config.py` (paths), `errors.py` (exception
types), `check.py` (end-to-end scoring), `cli.py` (commands), `evaluate.py` and `models.py`
(classifier comparison), `features.py` (hand features for the comparison models).

## Folders

| Folder | Contents |
|---|---|
| `media/eng/pdf/` | input PDFs |
| `data/gold/` | hand-labelled blocks (classifier training); rules in `LABELS.md` |
| `data/blocks/` | stage 1 output, one CSV (+ `.meta.json`) per PDF |
| `data/models/` | the trained classifier (regenerate with `dnd2json train`) |
| `data/output/` | one JSON per PDF, `_report.json` (status and warnings of the last run) |
| `data/expected/` | hand-written correct results for `dnd2json check` |

## When a PDF comes out wrong

1. Read its warnings in `data/output/_report.json` (also printed by `convert`).
2. Run `dnd2json convert "media/eng/pdf/<name>.pdf" --debug` and open
   `data/output/debug/<name>.debug.csv`: every block with its role, the rule that decided
   it (`role_reason`) and the classifier's P(description) (`p_desc`).
3. The wrong block shows which stage to fix: wrong text -> `clean.py`; wrong role ->
   `roles.py`; right roles but wrong JSON -> `structure.py`. Wording rules are in
   `patterns.py`: add the failing text to `tests/test_patterns.py` first, then fix.
4. To make the case permanent, add `data/expected/<name>.json` and run `dnd2json check`.

Errors: a PDF that can't be converted at all is reported as `failed` with the reason and
the batch continues. `error` (crash) means a bug: the traceback is in `_report.json`.

## Stage 1: extract blocks from PDFs

`convert` runs extraction automatically when a blocks CSV is missing or outdated. To run
it on its own (for example to OCR everything up front):

```
dnd2json extract              # text-layer PDFs here, scanned ones through .venv-ocr
dnd2json extract --text-only  # skip OCR
```

Outputs are skipped when they're up to date (same extractor version and settings, stored
in `<name>.meta.json`). Use `--force` to redo them.

## OCR setup

PaddlePaddle has no wheels for Python 3.14, so OCR runs in its own Python 3.12 env:

```
py -3.12 -m venv .venv-ocr
# NVIDIA GPU (CUDA 12.9 build; ~2.7 GB with the CUDA libraries):
.venv-ocr/Scripts/python -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
# or CPU only (much slower):
#   .venv-ocr/Scripts/python -m pip install paddlepaddle
.venv-ocr/Scripts/python -m pip install "paddleocr[doc-parser]>=3.3" pymupdf
```

Plain `pip install paddlepaddle` is the CPU-only build: it silently ignores the GPU.
`extract.py` picks the GPU automatically and warns when it falls back to CPU.

Models download on first use (a few hundred MB). On CPU, oneDNN is turned off in
`extract.py` because it crashes on some layout models in Paddle 3.3.

## Tests

`pytest` runs everything: wording rules (`test_patterns.py`), each stage on small
hand-made inputs (`test_clean.py`, `test_roles_structure.py`), the gold data
(`test_data.py`) and one real PDF end to end (`test_pipeline.py`, skipped without the PDF
or the trained model).
