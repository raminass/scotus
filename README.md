# SCOTUS_AID

## Description

This repository contains the code and data used in our paper titled "Expediting History: Using AI to Uncover Hidden Authorships of SCOTUS Opinions in Real Time". The paper is under review in Science and suggests AI model for Identifying Authorship in SCOTUS Opinions 1994-2023.

## Table of Contents

- [Usage](#usage)
- [Data](#data)
- [Code](#code)

## Usage

You can use the tool to identify author of opinions using [Visit SCOTUS_AID](https://raminass.github.io/SCOTUS_AI/)

## Data

Dataset used to train and evaluate the model: [raminass/opinions-94-23](https://huggingface.co/datasets/raminass/opinions-94-23)

## Code

The model reported in the paper and website, is the one trained with 17 justices, code and dependency packages used to collect and build the data `train_model.ipynb`.
Code used to run multiple models and select the best one `select_model.ipynb`, also each one of the experimented models provided within `m1-m10.ipynb`.
The best selected model is the `m4.ipynb`.
