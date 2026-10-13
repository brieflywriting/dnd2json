import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
SRD_df = pd.read_csv("./data/gold/5e-SRD-Combined.csv",).dropna(subset=["text"])
OCR_df=pd.read_csv("./data/gold/5e-OCR-Combined.csv").dropna(subset=["text"])

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