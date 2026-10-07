import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics for intrusion detection.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.

    y_pred : array-like
        Predicted labels.

    Returns
    -------
    dict
        Accuracy, Precision, Recall and F1.
    """

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def evaluate_model(model, dataloader, device="cpu"):
    """
    Evaluate a classification model on a dataset.

    The model is expected to return:
        logits, representation

    Parameters
    ----------
    model : torch.nn.Module
        Trained intrusion detection model.

    dataloader : torch.utils.data.DataLoader
        Evaluation data loader.

    device : str
        Device used for evaluation.

    Returns
    -------
    dict
        Classification metrics.
    """

    model.eval()

    all_labels = []
    all_predictions = []

    for features, labels in dataloader:

        features = features.to(device)
        labels = labels.to(device)

        logits, _ = model(features)

        predictions = torch.argmax(logits, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_predictions)
    )

    return metrics


def get_classification_report(y_true, y_pred):
    """
  
    """

    return classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
