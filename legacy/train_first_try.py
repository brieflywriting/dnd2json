from pathlib import Path

import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

pathToSubclasses = Path("./data/srd/5e-SRD-Subclasses.json")
pathToFeatures = Path("./data/srd/5e-SRD-Features.json")
pathToNewFeatures=Path("./data/gold/5e-OCR-CombinedNewClasses.csv")

SRD_df = pd.read_csv(pathToSubclasses).dropna(subset=["text"])
OCR_df=pd.read_csv(pathToFeatures).dropna(subset=["text"])
OCR_df_newclasses=pd.read_csv(pathToNewFeatures).dropna(subset=["text"])
#combined_df = pandas.concat([SRD_df, OCR_df], ignore_index=True)

real_train, real_test = train_test_split(
    OCR_df, test_size=0.25, stratify=OCR_df["label"], random_state=42)
train = pd.concat([real_train, SRD_df])
test = real_test

#from sklearn.dummy import DummyClassifier
#dummy_clf = DummyClassifier(strategy="most_frequent")
#dummy_clf.fit(train["text"], train["label"])
#print(dummy_clf.score(test["text"], test["label"]))

print(len(train), len(test),test["label"].value_counts())
model = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression(class_weight="balanced", max_iter=1000),
)
model.fit(train["text"], train["label"])
print(model.score(test["text"], test["label"]))
print("New classes: ")
print(model.score(OCR_df_newclasses["text"], OCR_df_newclasses["label"]))

def make_model():
    return make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    ) 


cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_idx, test_idx in cv.split(OCR_df["text"], OCR_df["label"], groups=OCR_df["doc_id"]):
    train = pd.concat([OCR_df.iloc[train_idx], SRD_df])
    test = OCR_df.iloc[test_idx]
    model = make_model().fit(train["text"], train["label"])
    scores.append(f1_score(test["label"], model.predict(test["text"]), average="macro"))
print(f"CV macro-F1: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
# final model on all OCR + SRD, scored once on the hold-out
model = make_model().fit(pd.concat([OCR_df, SRD_df])["text"], pd.concat([OCR_df, SRD_df])["label"])
print("New classes macro-F1:", f1_score(OCR_df_newclasses["label"], model.predict(OCR_df_newclasses["text"]), average="macro"))