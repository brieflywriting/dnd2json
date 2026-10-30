"""The block classifier: does a paragraph describe the subclass (class_description) or
state rules (feature)? Trained on the hand-labelled blocks in data/gold/.

Used by the pipeline:
    block_classifier.py   train on data/gold, save to data/models/, load, predict
                          (`dnd2json train`; stage 3b of the pipeline)

Research that picked the model (v0.1, see docs/PLAN.md sections 3-5):
    models.py             six candidate models, from a dummy baseline up to TF-IDF + LR
    features.py           hand-made features (dice, "you", level phrases) for some candidates
    evaluate.py           cross-validates every candidate the same way (`dnd2json evaluate`)

Training data:
    srd_to_csv.py         official SRD JSON (data/srd/) -> data/gold/5e-SRD-Combined.csv
"""
