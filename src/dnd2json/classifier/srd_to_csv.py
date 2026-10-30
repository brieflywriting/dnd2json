"""SRD JSON -> data/gold/5e-SRD-Combined.csv (text, label, doc_id, source).

Run:  python -m dnd2json.classifier.srd_to_csv

The CSV is generated, never hand-edited: change this script and rerun it instead.
"""

import pandas as pd

from dnd2json import config

SRD = config.SRD_DIR
OUT = config.GOLD_DIR / config.GOLD_SRD_FILE


def main():
    subclasses = pd.read_json(SRD / "5e-SRD-Subclasses.json")
    features = pd.read_json(SRD / "5e-SRD-Features.json")

    # Subclass intros. Description only: the one-line summary ("Channel Rage into Violent
    # Fury") is an SRD tagline that real PDFs don't have, and it used to be glued onto the
    # description without a space.
    class_df = pd.DataFrame({
        "text": subclasses["description"],
        "label": "class_description",
        "doc_id": "SRD - " + subclasses["name"],
    })

    # Features. Subclass features get their subclass as doc_id; base-class features
    # (Rage, Extra Attack, ...) have no subclass and get their class instead.
    subclass_name = features["subclass"].map(lambda s: s["name"] if isinstance(s, dict) else None)
    class_name = features["class"].map(lambda c: c["name"])
    features_df = pd.DataFrame({
        "text": features["description"],
        "label": "feature",
        "doc_id": "SRD - " + subclass_name.fillna(class_name),
    })

    combined = pd.concat([class_df, features_df], ignore_index=True)
    combined["source"] = "srd"
    combined["text"] = combined["text"].str.strip()
    combined = combined[combined["text"].str.len() > 0]

    combined.to_csv(OUT, index=False, encoding="utf-8")
    print(f"Wrote {len(combined)} rows to {OUT}")
    print(combined["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
