from transformers import pipeline
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch.nn as nn
import torch
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import evaluate

# https://huggingface.co/spaces/evaluate-metric/accuracy
accuracy = evaluate.load("accuracy")
tokenizer = AutoTokenizer.from_pretrained(
    "nlpaueb/legal-bert-small-uncased", use_fast=True
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def load_trained_model(name, k=13):
    cls = pipeline(
        "text-classification", model=name, top_k=k, padding=True, truncation=True
    )
    return cls


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


def predict_labels(inputs, cls):
    all_predictions = []
    for text in inputs:
        inputs = cls.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        logits = cls.model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        all_predictions.append(predicted_class_id)

    return all_predictions


# https://discuss.huggingface.co/t/how-can-i-use-class-weights-when-training/1067/3
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        label_w = [
            1.03,
            1.07,
            1.0,
            2.5,
            3.59,
            1.12,
            1.98,
            1.36,
            1.2,
            2.41,
            1.47,
            2.03,
            4.2,
        ]
        loss_fct = nn.CrossEntropyLoss(
            weight=torch.tensor(label_w, device=model.device)
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def tokenize_dataset(ds):
    tokenized_court = ds.map(preprocess_function, batched=True)
    return tokenized_court


def get_new_trainer(
    id2label,
    label2id,
    tokenized_court,
    freeze=False,
    epochs=20,
    model_name="sc",
    batch_size=64,
):
    model = AutoModelForSequenceClassification.from_pretrained(
        "nlpaueb/legal-bert-small-uncased",
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    if freeze:
        for param in model.bert.parameters():
            param.requires_grad = False

    training_args = TrainingArguments(
        f"{model_name}",
        # logging_dir=f"legal-bert-small-uncased-finetuned-{task}/runs/1",
        learning_rate=2e-5,  # as in bert paper
        per_device_train_batch_size=batch_size,  # in legal-bert 256 however no memory here
        per_device_eval_batch_size=batch_size,  # in legal-bert 256 however no memory here
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=True,
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
