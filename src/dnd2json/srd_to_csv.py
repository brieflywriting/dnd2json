from importlib.resources import path
from pathlib import Path
import pandas

pathToSubclasses = Path("./data/srd/5e-SRD-Subclasses.json")

#extract class descriptions
jsonClasses=pandas.read_json(pathToSubclasses)
class_columns=jsonClasses["description"]
class_df=pandas.DataFrame({"text": class_columns, "label": "class_description"})

#extract feature descriptions
pathToFeatures = Path("./data/srd/5e-SRD-Features.json")
jsonFeatures=pandas.read_json(pathToFeatures)
features_columns=jsonFeatures["description"]
features_df=pandas.DataFrame({"text": features_columns, "label": "feature"})

#concat into one dataset
combined_df=pandas.concat([class_df, features_df], ignore_index=True)


combined_df.to_csv("./data/gold/5e-SRD-Combined.csv", index=False)


#one type of data is class:
# iter, type, doc_id
#the other is a feature
#only name and description