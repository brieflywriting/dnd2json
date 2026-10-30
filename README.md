# dnd2json

Converts D&D 5e homebrew subclass PDFs into structured JSON: subclass name, class,
description and features (name, level, rules text), plus tables, stat blocks and lore.

```
dnd2json train                  # once: train the block classifier on data/gold
dnd2json convert --all          # every PDF in media/eng/pdf -> data/output/*.json
dnd2json check                  # score the output against data/expected
```

(`dnd2json` is installed by `pip install -e ".[dev]"`. If the command isn't found,
`python -m dnd2json.cli train` etc. does the same.)

## Project map

```
dnd2json/
├── README.md                   how to run, project map, debugging (this file)
├── pyproject.toml              package metadata, dependencies, the `dnd2json` command
│
├── src/dnd2json/               the Python package
│   ├── cli.py                  the commands: train, convert, check, extract, evaluate
│   ├── config.py               every path and file name, in one place
│   ├── errors.py               exception types
│   │
│   ├── pipeline/               PDF -> JSON: the product
│   │   ├── convert.py          runs stages 1-5 for one PDF or a batch, writes _report.json
│   │   ├── extract.py          1   PDF -> blocks CSV (text layer or OCR)
│   │   ├── clean.py            2   reading order, junk, drop caps, OCR repairs
│   │   ├── roles.py            3a  heading / body / table / level_line / other
│   │   ├── structure.py        4   sections -> subclasses, features, levels
│   │   ├── schema.py           5   the output format (pydantic)
│   │   ├── patterns.py         wording rules (regexes) used by stages 2-4
│   │   └── check.py            scores data/output/ against data/expected/
│   │
│   └── classifier/             "is this paragraph a description or rules?" (ML)
│       ├── block_classifier.py 3b  the model the pipeline uses: train, save, load, predict
│       ├── models.py           six candidate models compared in v0.1
│       ├── features.py         hand-made features used by some candidates
│       ├── evaluate.py         cross-validates the candidates
│       └── srd_to_csv.py       SRD JSON -> SRD training CSV in data/gold/
│
├── tests/                      pytest, one file per area (see "Tests")
│
├── data/
│   ├── gold/                   hand-labelled blocks: classifier training data
│   ├── srd/                    official SRD JSON, the source of the SRD gold CSV
│   ├── recheck/                blind re-labelling sample (PLAN 2.6)
│   ├── blocks/                 stage 1 cache: <pdf>.csv + <pdf>.meta.json
│   ├── output/                 the result: <pdf>.json + _report.json (last run's status)
│   ├── expected/               hand-written correct results for `dnd2json check`
│   └── models/                 the trained classifier (not in git: `dnd2json train`)
│
├── docs/
│   ├── PLAN.md                 steps done and to do, each with a check
│   ├── NEXT.md                 the one step to do next
│   ├── DECISIONS.md            why things are built this way
│   └── LABELS.md               labelling rules for data/gold
│
├── legacy/                     old scripts kept for reference; nothing imports them
└── media/eng/pdf/              the input PDFs (not in git)
```

### Which file do I open?

| I want to... | Open |
|---|---|
| add or change a command | `cli.py` |
| move a folder, rename a data file | `config.py` |
| change how text comes out of a PDF, OCR settings | `pipeline/extract.py` |
| fix broken text (OCR misreads, line breaks, drop caps) | `pipeline/clean.py` |
| teach it a new wording ("Beginning at 3rd level", credits lines) | `pipeline/patterns.py` + `tests/test_patterns.py` |
| fix what counts as a heading, body text, table | `pipeline/roles.py` |
| fix how blocks become features, levels, sub-options | `pipeline/structure.py` |
| change the JSON format | `pipeline/schema.py` |
| change error handling, warnings, the batch report | `pipeline/convert.py` |
| retrain the classifier, change what it learns from | `classifier/block_classifier.py` |
| try a new model against the old ones | `classifier/models.py`, then `dnd2json evaluate` |

## The pipeline

```
PDF ─► 1 extract ─► 2 clean ─► 3 roles + classifier ─► 4 structure ─► 5 JSON
       blocks CSV    order,       heading/body/table/     sections ->     validated
       per PDF       junk,        level_line/other,       subclasses,     (schema.py)
                     drop caps    P(description)          features
```

| Stage | File | What it does |
|---|---|---|
| 1 | `pipeline/extract.py` | PDF -> one row per text block. Text-layer PDFs via PyMuPDF (font size, bold, italic); scanned or garbled PDFs via PaddleOCR (layout labels, confidence). Cached in `data/blocks/`. |
| 2 | `pipeline/clean.py` | Column reading order, junk removal, drop caps ("C"+"haotic"), OCR digit fixes ("1Oth"), line unwrapping, level lines split off paragraphs. |
| 3a | `pipeline/roles.py` | Each block's layout role, with the rule that decided it (`role_reason`). |
| 3b | `classifier/block_classifier.py` | TF-IDF + logistic regression (`words_only` from `models.py`): is body text a description or rules? |
| 4 | `pipeline/structure.py` | Groups blocks into sections under headings and decides what each one is: description, feature (+ level), sub-option, table, stat block, spell, lore. Splits files holding several subclasses. |
| 5 | `pipeline/schema.py` | The output format (pydantic): documents what the JSON contains and validates it. |

`pipeline/convert.py` runs the stages in this order; each module's docstring explains its
stage in detail.

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
`pipeline/extract.py` picks the GPU automatically and warns when it falls back to CPU.
It imports nothing from the package, so the OCR env runs it as a plain script.

Models download on first use (a few hundred MB). On CPU, oneDNN is turned off in
`extract.py` because it crashes on some layout models in Paddle 3.3.

## Tests

`pytest` runs everything:

| Test file | Checks |
|---|---|
| `tests/test_patterns.py` | the wording rules in `pipeline/patterns.py`, on real snippets |
| `tests/test_clean.py` | stage 2 on small hand-made tables |
| `tests/test_roles_structure.py` | stages 3a and 4 on hand-made blocks |
| `tests/test_pipeline.py` | one real PDF end to end, the JSON format, error handling (skipped without the PDF or the trained model) |
| `tests/test_features.py` | the hand features in `classifier/features.py` |
| `tests/test_data.py` | the gold CSVs: encoding, labels, doc_ids, duplicates |
