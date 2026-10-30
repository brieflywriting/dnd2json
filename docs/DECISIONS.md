# Decisions

Why the project is built the way it is. Newest first within each section. When you change
one of these, edit the entry and say why, so the reasoning stays traceable.

## Code layout

- **Two subpackages by purpose:** `pipeline/` is the product (PDF -> JSON, one module
  per stage), `classifier/` is the machine-learning part (the model the pipeline uses,
  plus the research that picked it). They meet in one line: `pipeline/convert.py`
  loads `classifier/block_classifier.py`. Shared basics (`cli`, `config`, `errors`)
  stay at the top.
- **`__init__.py` files hold only documentation**, no re-exports. Imports name the full
  module path, so you can always see where a function lives, and `pipeline` and
  `classifier` can import each other without import cycles.
- **`pipeline/extract.py` imports nothing from the package**, because the OCR env
  (`.venv-ocr`, Python 3.12) runs it as a plain script.
- **Old code lives in `legacy/`**, outside `src/`, so it's never installed or imported.

## Data

- **Labels** (PLAN 1.1): `class_description`, `feature`, `other`. Rules and borderline
  cases in `LABELS.md`. Content decides the label, not position or pronoun.
- **Feature granularity** (PLAN 1.2): in the gold CSVs, a `feature` block is one paragraph
  of a feature's text, as it comes out of the PDF. In the pipeline, a feature is assembled
  from a heading + level line + all paragraphs until the next heading (`structure.py`).
- **Data contract** (PLAN 1.3): gold CSVs have `text, label, doc_id`. `source` (`real` /
  `srd`) is added when loading (`evaluate.py`) or written by `srd_to_csv.py`. There is no
  `block_idx` in the gold data yet: block order was lost when the blocks were labelled.
  The pipeline's own blocks (`data/blocks/`) do have `block_idx`, `page` and font data.
- **`other` blocks** aren't learned by the classifier (only 1 gold example). Page
  furniture is recognised by rules in `roles.py` instead.
- **SRD data** is training-only, never scored. It's shaped differently from real PDFs
  (taglines, base-class features), and it hurt the hand-feature models; `words_only` is
  unaffected. Kept for now, pending the PLAN 5.1 ablation.
- **Hold-out:** `5e-OCR-CombinedNewClasses.csv` was meant for a one-time final test, but it
  has been scored several times and its labels were fixed after seeing model errors. It's
  dev data now. A fresh hold-out must come from new PDFs (PLAN 9.4).

## Classifier

- **Model:** `words_only` = word TF-IDF (1–2 grams, sublinear tf, min_df 2) + logistic
  regression (balanced classes). It beat dummy, rules, hand features and combinations:
  macro-F1 0.991 (1 error in 135) in document-grouped CV.
- **Honest estimate is lower (~0.95):** grouped by class (train on two of Bard/Rogue/Wizard,
  test on the third) the score drops, because words like "college" and "bards" leak the
  class. Fix planned: class-grouped CV and masking class/subclass names (PLAN 9.1, 9.3).
- **Hand features stay out of the model** for now: on a standardised scale they overpower
  the word weights, and SRD distorts them. They may matter once labels differ by shape
  (titles, tables).
- **Target** (PLAN 1.4): not written down when set. Write it here.

## Extraction (stage 1)

- **Two paths, decided per PDF:** a usable text layer -> PyMuPDF (exact text, fonts);
  garbled or scanned -> PaddleOCR PP-StructureV3. Per PDF, not per page, so all blocks of
  a document share one shape. 27 of 59 PDFs take the text path.
- **OCR runs in a separate Python 3.12 venv** (`.venv-ocr`): PaddlePaddle has no Python
  3.14 wheels. The pipeline calls it as a subprocess.
- **GPU build of Paddle** (`paddlepaddle-gpu`, CUDA 12.9): the CPU build took >1 h for the
  OCR documents, the GPU 52 s. Settings: 150 dpi, native-size detection, table structure
  recognition off (table text comes from the OCR lines), English v5 recognition model.
- **Text-path blocks are split at font changes**, so headings glued to paragraphs become
  separate blocks; Homebrewery's duplicated heading lines are dropped.

## Pipeline (stages 2–5)

- **Layout first, words second:** roles (heading, body, table, level line, other) come
  from layout rules; the classifier only judges body text (description vs rules).
- **Every rule records why it fired** (`role_reason`, `--debug` CSV), so wrong output can be
  traced to a stage.
- **Reading order is rebuilt** for text-layer pages (column by column); OCR keeps the
  layout model's order.
- **Level sources, most reliable first:** level line ("3rd-level X feature") / level in the
  heading -> a sentence starting with the grant ("At 3rd level, ...") -> the
  feature-summary table. Features are sorted by level (OCR can scramble page order).
- **Sub-options** ("Mask of Comedy" under "Channel the Muse") are merged into the feature
  above when the document announces every feature's level and the heading has none, or a
  confirmed summary table doesn't list it.
- **One file, several subclasses** only when the filename says so ("&", "and", plural
  "Schools of"); the document is then split at each subclass title.
- **Soft problems are warnings, hard problems are exceptions:** a JSON is still written
  with warnings ("no level found for X"); a PDF that can't be converted is reported as
  failed and the batch continues.
- **Output format** mirrors the SRD API's field names (`name`, `level`, `description`), so
  homebrew and official subclasses can be read by the same code.
- **End-to-end check** (`dnd2json check`) uses hand-written expected files written from
  the PDFs, never from the pipeline's output.
