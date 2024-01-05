from utils import *
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import matplotlib.pyplot as plt


def plot_confusion(y_test, y_pred, label2id):
    cm = confusion_matrix(y_test, y_pred)
    cmd = ConfusionMatrixDisplay(
        cm,
        display_labels=list(label2id.keys()),
    )
    cmd.plot(xticks_rotation="vertical")


def print_metrics(y_test, y_pred):
    print("\n-------------------- Key Metrics --------------------")
    print("\nAccuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))
    print("Balanced Accuracy: {:.2f}\n".format(balanced_accuracy_score(y_test, y_pred)))

    print(
        "Micro Precision: {:.2f}".format(
            precision_score(y_test, y_pred, average="micro")
        )
    )
    print("Micro Recall: {:.2f}".format(recall_score(y_test, y_pred, average="micro")))
    print("Micro F1-score: {:.2f}\n".format(f1_score(y_test, y_pred, average="micro")))

    print(
        "Macro Precision: {:.2f}".format(
            precision_score(y_test, y_pred, average="macro")
        )
    )
    print("Macro Recall: {:.2f}".format(recall_score(y_test, y_pred, average="macro")))
    print("Macro F1-score: {:.2f}\n".format(f1_score(y_test, y_pred, average="macro")))

    print(
        "Weighted Precision: {:.2f}".format(
            precision_score(y_test, y_pred, average="weighted")
        )
    )
    print(
        "Weighted Recall: {:.2f}".format(
            recall_score(y_test, y_pred, average="weighted")
        )
    )
    print(
        "Weighted F1-score: {:.2f}".format(f1_score(y_test, y_pred, average="weighted"))
    )

    print("\n--------------- Classification Report ---------------\n")
    print(classification_report(y_test, y_pred))
    


def numpy_acc(predictions, labels):
    predictions = np.argmax(predictions, axis=1)
    # Calculate accuracy
    accuracy = (predictions == labels).mean()
    return accuracy
