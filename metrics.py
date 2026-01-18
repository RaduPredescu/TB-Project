import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def compute_classification_metrics(y_true, y_pred):
    """
    Compute standard classification metrics.

    Returns a dict with:
      - accuracy
      - precision (macro)
      - recall (macro)
      - f1 (macro)
      - confusion_matrix
    """
    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


def print_metrics(name, metrics):
    print(f"\n=== {name} ===")
    print(f"Accuracy : {metrics['accuracy'] * 100:.2f} %")
    print(f"Precision: {metrics['precision'] * 100:.2f} %")
    print(f"Recall   : {metrics['recall'] * 100:.2f} %")
    print(f"F1-score : {metrics['f1'] * 100:.2f} %")
    print("Confusion matrix:\n", metrics["confusion_matrix"])
