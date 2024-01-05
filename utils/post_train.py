from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from utils import *


def load_model_from_disk(path, top_k=9):
    # Path to your local checkpoint directory
    local_checkpoint_path = path

    # Load the model from the local checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(local_checkpoint_path)

    # Load the tokenizer from the local checkpoint
    tokenizer = AutoTokenizer.from_pretrained(local_checkpoint_path)

    # Create a pipeline with the loaded model and tokenizer
    text_classification_pipeline = pipeline(
        "text-classification",  # or any other task you're interested in
        model=model,
        tokenizer=tokenizer,
        padding=True,
        truncation=True,
        top_k=top_k,
    )


def load_model_from_hub(name, top_k=9):
    cls = pipeline(
        "text-classification", model=name, top_k=top_k, padding=True, truncation=True
    )
    return cls


def predict_labels(inputs, cls):
    all_predictions = []
    for text in inputs:
        inputs = cls.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        logits = cls.model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        all_predictions.append(predicted_class_id)

    return all_predictions


def average_bulk(result):
    pred = {}
    for c in result:
        for d in c:
            if d["label"] not in pred:
                pred[d["label"]] = [round(d["score"], 2)]
            else:
                pred[d["label"]].append(round(d["score"], 2))
    sumary = {k: round(sum(v) / len(v), 2) for k, v in pred.items()}
    sorted_dict = dict(sorted(sumary.items(), key=lambda x: x[1], reverse=True))
    final = ""
    for x in list(sorted_dict)[0:3]:
        final += f"""{x}:{sorted_dict[x]}, """
    return final, list(sorted_dict)[0]


def average_text(text, model, judges):
    result = model(text)
    new_res = []
    for d in result:
        p = {}
        for dicts in d:
            if dicts["label"] in judges:
                p[dicts["label"]] = dicts["score"]
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


def average_opinion(result):
    pred = {}
    for c in result:
        for d in c:
            if d["label"] not in pred:
                pred[d["label"]] = [round(d["score"], 2)]
            else:
                pred[d["label"]].append(round(d["score"], 2))
    sumary = {k: round(sum(v) / len(v), 2) for k, v in pred.items()}
    sorted_dict = dict(sorted(sumary.items(), key=lambda x: x[1], reverse=True))
    final = ""
    for x in list(sorted_dict)[0:3]:
        final += f"""{x}:{sorted_dict[x]}, """
    return final, list(sorted_dict)[0]


tokenizer = AutoTokenizer.from_pretrained(
    "nlpaueb/legal-bert-small-uncased", use_fast=True, cache_dir="./.cache"
)


def get_pipe(model):
    return pipeline(
        "text-classification",
        model=model,
        padding=True,
        truncation=True,
        tokenizer=tokenizer,
    )
