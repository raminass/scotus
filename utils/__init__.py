from .cleaning import remove_citations, split_data, split_text, chunk_data
import pandas as pd
import numpy as np
import json
from IPython.display import display, HTML


def get_labels_maping(df):
    judges_list = df["author_name"].unique().tolist()
    # map judge name to index
    judge2idx = {judge: idx for idx, judge in enumerate(judges_list)}
    # map index to judge name
    idx2judge = {idx: judge for idx, judge in enumerate(judges_list)}
    return idx2judge, judge2idx


def find_case_by_name(df, name):
    return display(
        HTML(
            df[df["case_name"].str.contains(name)]
            .iloc[:, :-1]
            .to_html(render_links=True, escape=False)
        )
    )


def normaliz_dict(d, target=1.0):
    raw = sum(d.values())
    factor = target / raw
    return {key: value * factor for key, value in d.items()}


def average_text(text, model, judges):
    result = model(text)
    new_res = []
    for d in result:
        p = {}
        for dicts in d:
            if dicts["label"] in judges:
                p[dicts["label"]] = round(dicts["score"], 2)
        p = normaliz_dict(p)
        new_res.append(p)

    pred = {}
    for c in new_res:
        for k, v in c.items():
            if k not in pred:
                pred[k] = [round(v, 2)]
            else:
                pred[k].append(round(v, 2))
    sumary = {k: round(sum(v) / len(v), 2) for k, v in pred.items()}
    sumary = normaliz_dict(sumary)
    return dict(sorted(sumary.items(), key=lambda x: x[1], reverse=True)), new_res
