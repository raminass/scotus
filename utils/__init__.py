from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import numpy as np

# import json

# with open("data/id2label.json", "r") as j:
#     id2label = json.loads(j.read())

# with open("data/label2id.json", "r") as j:
#     label2id = json.loads(j.read())


def normaliz_dict(d, target=1.0):
    raw = sum(d.values())
    factor = target / raw
    return {key: value * factor for key, value in d.items()}
