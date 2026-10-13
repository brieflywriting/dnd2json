# Plan: block classifier (v0.1)

**Goal:** given one block of text from a subclass document, predict what it is (title, flavor, feature, ...).
**Out of scope for now:** the divider (blocks are split by hand), field extraction, JSON output, OCR, GUI.
**Done when:** `dnd2json classify blocks.csv` writes a label + confidence per block, and macro-F1 on the held-out real documents reaches the target you set in step 1.4.

Each step is 10–30 min and ends with a check. Tick it, commit, update `NEXT.md`.

## 0. Setup (once)
- [x] 0.1 `git init`, `.gitignore` (`.venv/`, `media/`, `*.egg-info/`, `__pycache__/`), first commit. *Check:* `git status` is clean.
- [x] 0.2 Remove `"paddle"` from `pyproject.toml`, `pip install -e ".[dev]"`. *Check:* `pytest` runs (0 tests is fine).
- [x] 0.3 Folders: `data/srd/` (move `raw_data/*` here), `data/gold/`, `data/synthetic/`. *Check:* `raw_data/` is gone.

## 1. Define the problem
- [ ] 1.1 `LABELS.md`: each label, one-sentence definition, two real examples. Include `unknown`. *Check:* you can label 10 random blocks without hesitating.
- [ ] 1.2 Decide feature granularity: one `feature` block (heading + level + body) or separate blocks. Write it in `DECISIONS.md`.
- [ ] 1.3 Data contract: CSV columns `doc_id, block_idx, text, label, source` (`source` = `srd` or `real`). Write it in `DECISIONS.md`.
- [ ] 1.4 Set a target number: macro-F1 = ? on real documents.

## 2. Data
- [ ] 2.1 `srd_to_csv.py`: SRD subclasses → `data/synthetic/srd_blocks.csv`. *Check:* 12 docs, Berserker reads in order.
- [ ] 2.2 Labeling format: copy a real `.md` into `data/gold/`, put a line `@label` above each block (blank line between blocks). *Check:* one file done by hand.
- [ ] 2.3 `gold_to_csv.py`: parse the `@label` files → `data/gold/gold_blocks.csv`, plus a pytest with a 3-block example. *Check:* test passes.
- [ ] 2.4 Label 5 real docs from different formats (Devout Inquisitor, Marksman, Sirens, Transmutation, Shadowfell). *Check:* label counts printed; every label has ≥ 3 examples.
- [ ] 2.5 Label 5 more (aim: ~150 blocks, ≥ 10 per label; hunt for rare ones like `credits`, `table`).

## 3. Evaluation harness (before any real model)
- [ ] 3.1 `evaluate.py`: load CSV, `GroupKFold` by `doc_id`, print `classification_report` + confusion matrix + macro-F1.
- [ ] 3.2 Baseline: `DummyClassifier(strategy="most_frequent")`. *Check:* a number to beat.

## 4. Features and models
- [ ] 4.1 Rules baseline: a few `if`s (starts with `3rd-level` → feature; short + no period → title/heading; ...). Run through the same harness.
- [ ] 4.2 Hand features: length, line count, ends with period, digit and capital fraction, level-pattern match, has `**X.**` run-in, second person ("you") vs third person, relative position in doc, table-ish (many short lines / digits). Test the level pattern with pytest.
- [ ] 4.3 Logistic regression (`class_weight="balanced"`) on hand features. Compare with 3.2 and 4.1.
- [ ] 4.4 Add TF-IDF (words, then `char_wb` 2–5) via `ColumnTransformer`. Compare again.

## 5. Real evaluation
- [ ] 5.1 Ablation: train on real only vs real + SRD; score on real only.
- [ ] 5.2 Error analysis: read every misclassified block, write the pattern in `DECISIONS.md`, fix features or labels (not the test set).
- [ ] 5.3 Hold-out run: one format never seen in training. Look at it once.

## 6. Make it usable
- [ ] 6.1 `predict(blocks) -> [(label, confidence)]`; below threshold → `unknown`.
- [ ] 6.2 Save the model with joblib, load it back, same predictions.
- [ ] 6.3 CLI: `dnd2json classify in.csv -o out.csv`. Tag `v0.1-classifier`.

## Parked
Divider, field extraction, JSON schema, OCR fixes, linking, search, GUI, other entity types.
