# Plan

Each step is 10–30 min and ends with a check. Tick it, commit, update `NEXT.md`.
Why things are the way they are: `DECISIONS.md`. How to run, project map, debugging:
`README.md` in the repo root.

Status: **v0.1 (classifier) done, v0.2 (working PDF -> JSON pipeline) done.**
Next: v0.3, make the numbers honest and the pipeline robust (sections 8–10).

---

# v0.1 Block classifier ✅

**Goal:** given one block of text from a subclass document, predict what it is.
**Outcome:** `words_only` (TF-IDF + logistic regression), macro-F1 0.991 in document-grouped
CV, ~0.95 when a whole class is held out. Used as stage 3b of the pipeline.

## 0. Setup (once)
- [x] 0.1 `git init`, `.gitignore`, first commit.
- [x] 0.2 Remove `"paddle"` from `pyproject.toml`, `pip install -e ".[dev]"`.
- [x] 0.3 Folders: `data/srd/`, `data/gold/`, `data/synthetic/` (removed in 7.11: unused).

## 1. Define the problem
- [x] 1.1 `LABELS.md`: labels, definitions, examples, ordered rules, 8 borderline cases.
- [x] 1.2 Feature granularity -> `DECISIONS.md`.
- [x] 1.3 Data contract -> `DECISIONS.md`. (`block_idx` is missing from the gold data: see 9.4.)
- [x] 1.4 Target macro-F1. *Not written down: add it to `DECISIONS.md`.*

## 2. Data
- [x] 2.1 `srd_to_csv.py`: SRD -> `data/gold/5e-SRD-Combined.csv` (with `doc_id`, `source`).
- [x] 2.2–2.5 Real documents labelled: 135 + 78 blocks (`5e-OCR-Combined*.csv`).
- [x] 2.6 Blind re-check: 30/30 agreement (mostly easy feature blocks: see 9.4).
- [x] 2.7 Relabels from `LABELS.md` applied, `�` characters repaired, data tests
      (`tests/test_data.py`). *One open item: row 27 of NewClasses has `???` (8.3).*

## 3. Evaluation harness
- [x] 3.1 `evaluate.py`: `StratifiedGroupKFold` by `doc_id`, 5 seeds, report + confusion
      matrix + macro-F1 + error counts.
- [x] 3.2 Baseline `dummy`: macro-F1 0.41.

## 4. Features and models
- [x] 4.1 `rules`: 0.93.
- [x] 4.2 Hand features (`features.py`). Position in document skipped: no `block_idx`.
- [x] 4.3 `hand_lr`: 0.89.
- [x] 4.4 `hand+words` 0.92, `hand+words+chars` 0.94, **`words_only` 0.99** (winner).

## 5. Real evaluation
- [ ] 5.1 SRD ablation, done properly. -> moved to **9.2**.
- [x] 5.2 Error analysis: label fixes and decisions recorded in `LABELS.md` / `DECISIONS.md`.
- [–] 5.3 Hold-out run: the NewClasses file was scored too often to count. -> **9.4**.

## 6. Make it usable
- [–] 6.1 Confidence threshold -> `unknown`. -> moved to **9.5** (needs calibration).
- [x] 6.2 Model saved/loaded with joblib (`classifier/block_classifier.py`, `dnd2json train`).
- [x] 6.3 CLI. Became `dnd2json convert` (PDF -> JSON) instead of `classify` (CSV -> CSV).

---

# v0.2 Pipeline: PDF -> JSON ✅

Built "walking skeleton" first, then each stage improved on all 59 PDFs.

## 7. Stages
- [x] 7.1 Stage 1 `extract.py`: text layer (PyMuPDF, fonts) or OCR (PaddleOCR, GPU).
      *Check:* 59/59 PDFs -> `data/blocks/` (27 text, 32 OCR, OCR in 52 s).
- [x] 7.2 Stage 2 `clean.py`: column order, junk, drop caps, OCR repairs, level-line split.
- [x] 7.3 Stage 3a `roles.py`: heading / body / table / level_line / other, with reasons.
- [x] 7.4 Stage 3b `block_classifier.py`: train, save, load, P(description).
- [x] 7.5 Stage 4 `structure.py`: sections, levels, sub-options, stat blocks, spells,
      summary tables, multi-subclass files.
- [x] 7.6 Stage 5 `schema.py`: validated JSON (SRD-style field names).
- [x] 7.7 `pipeline/convert.py` + `errors.py`: batch never stops on one PDF, `_report.json`.
- [x] 7.8 `cli.py`: `train`, `convert`, `check`, `extract`, `evaluate`.
- [x] 7.9 `check.py` + 8 expected files. *Check:* recall 95%, precision 100%, levels 100%.
- [x] 7.10 Tests: 93 passing (`pytest`).
      *Check:* `dnd2json convert --all` -> 59 PDFs, 51 ok, 8 warnings, 0 failed.
- [x] 7.11 Project layout: modules grouped into `src/dnd2json/pipeline/` (PDF -> JSON) and
      `src/dnd2json/classifier/` (ML), docs in `docs/`, old code in `legacy/`. Map in the
      README. *Check:* `pytest` green, `convert` works on a text and an OCR PDF.

---

# v0.3 Honest numbers, robust pipeline

## 8. Wrap up v0.2
- [ ] 8.1 Read the 8 files in `data/expected/` against the PDFs: they're the ground truth
      for `check`. *Check:* you'd sign every name and level.
- [ ] 8.2 Commit, tag `v0.2-pipeline`. *Check:* `git status` clean, `pytest` green.
- [ ] 8.3 Fix row 27 of `5e-OCR-CombinedNewClasses.csv` (`???`) from the PDF.
- [ ] 8.4 Lint `classifier/features.py` (`ruff check src tests`). *Check:* 0 errors.
      (`train.py` moved to `legacy/train_first_try.py`: `evaluate.py` replaced it.)

## 9. Make the classifier numbers honest (from the methodology review)
- [ ] 9.1 Leave-one-class-out CV in `classifier/evaluate.py` (group by class, not document).
      *Check:* the score words_only gets when Bard/Rogue/Wizard is unseen.
- [ ] 9.2 SRD ablation: words_only with vs. without SRD, on 9.1's CV. Decide, log it.
- [ ] 9.3 Mask class and subclass names before TF-IDF ("bards", "college" -> placeholders).
      *Check:* 9.1 score goes up, or stays and the top words stop being class names.
- [ ] 9.4 Fresh hold-out: label 5 new PDFs of other classes (Paladin, Sorcerer...) from
      `data/blocks/` (they have `block_idx` and `other` blocks). Score once, log it.
- [ ] 9.5 Calibration: are P=0.8 predictions right 80% of the time? Pick an `unknown`
      threshold inside CV.

## 10. Pipeline quality
- [ ] 10.1 Grow `data/expected/` to ~15 documents, mostly OCR. *Check:* `dnd2json check`.
- [ ] 10.2 OCR headings filed as page headers (Mimic Magic's "Mana Devouring"): place
      blocks without reading order by their position on the page.
- [ ] 10.3 Text boxes read as images (Command's "Gallant Leader"): split the heading and
      level line out of the rescued text, drop the " n " artefacts.
- [ ] 10.4 Drop caps lost by OCR ("he College of Flavour", "ou are one of"): restore the
      first letter when the word is unknown.
- [ ] 10.5 Tables as rows and cells instead of plain text.

## Parked
GUI, linking to official content, search, other entity types (spells, feats, monsters),
non-English PDFs.
