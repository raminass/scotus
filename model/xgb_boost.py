import xgboost as xgb


from sklearn.utils.class_weight import compute_sample_weight

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def xgb_train_new(X_train, X_test, y_train, y_test):
    xgb_clf = xgb.XGBClassifier(
        objective="multi:softmax",
        booster="gblinear",
        # reg_lambda = 0.1,
        # objective='multi:softprob',
        num_class=13,
        # missing=0,
        # gamma=1, # default gamma value
        # learning_rate=0.1,
        max_depth=5,  # re-optimized from v2
        # reg_lambda=1, # default L2 value
        # subsample=0.8, # tried but not ideal
        # colsample_bytree=0.3, # tried but not ideal
        # early_stopping_rounds=10,
        eval_metric=["merror", "mlogloss", "auc"],
        seed=42,
    )

    xgb_clf.fit(
        X_train,
        y_train,
        verbose=0,  # set to 1 to see xgb training round intermediate results
        sample_weight=compute_sample_weight(class_weight="balanced", y=y_train),  # class weights to combat unbalanced 'target'
        eval_set=[(X_train, y_train), (X_test, y_test)],
    )
    return xgb_clf


def xgb_plot_results(xgb_clf):
  # preparing evaluation metric plots
    results = xgb_clf.evals_result()
    epochs = len(results['validation_0']['mlogloss'])
    x_axis = range(0, epochs)
    # xgboost 'mlogloss' plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_axis, results["validation_0"]["mlogloss"], label="Train")
    ax.plot(x_axis, results["validation_1"]["mlogloss"], label="Test")
    ax.legend()
    plt.ylabel("mlogloss")
    plt.title("GridSearchCV XGBoost mlogloss")
    plt.show()
    # xgboost 'merror' plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_axis, results["validation_0"]["merror"], label="Train")
    ax.plot(x_axis, results["validation_1"]["merror"], label="Test")
    ax.legend()
    plt.ylabel("merror")
    plt.title("GridSearchCV XGBoost merror")
    plt.show()
    # xgboost 'merror' plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_axis, results["validation_0"]["auc"], label="Train")
    ax.plot(x_axis, results["validation_1"]["auc"], label="Test")
    ax.legend()
    plt.ylabel("auc")
    plt.title("GridSearchCV XGBoost AUC")
    plt.show()
