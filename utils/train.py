from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)
import torch.nn as nn
import torch
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import evaluate


pre_trained = "nlpaueb/legal-bert-small-uncased"
# pre_trained = "raminass/scotus-v10"
# https://huggingface.co/spaces/evaluate-metric/accuracy
accuracy = evaluate.load("accuracy")
tokenizer = AutoTokenizer.from_pretrained(
    pre_trained, use_fast=True, cache_dir="./.cache"
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


def tokenize_dataset(ds):
    tokenized_court = ds.map(preprocess_function, batched=True)
    return tokenized_court


def get_trainer(
    id2label,
    label2id,
    tokenized_court,
    freeze=False,
    epochs=5,
    model_name="sc",
    batch_size=16,
    push_to_hub=False,
):
    model = AutoModelForSequenceClassification.from_pretrained(
        pre_trained,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        cache_dir="./.cache",
        ignore_mismatched_sizes=True,
    )

    if freeze:
        for param in model.bert.parameters():
            param.requires_grad = False

    training_args = TrainingArguments(
        f"{model_name}",
        # logging_dir=f"legal-bert-small-uncased-finetuned-{task}/runs/1",
        overwrite_output_dir=True,
        learning_rate=2e-5,  # as in bert paper
        per_device_train_batch_size=batch_size,  # in legal-bert 256 however no memory here
        per_device_eval_batch_size=batch_size,  # in legal-bert 256 however no memory here
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=push_to_hub,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_court["train"],
        eval_dataset=tokenized_court["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    return trainer


def load_trained_model(name, k=13):
    return pipeline(
        "text-classification", model=name, top_k=k, padding=True, truncation=True
    )


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
