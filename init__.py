import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class IntrusionDataset(Dataset):
    """
    PyTorch dataset for intrusion detection data.
    """

    def __init__(self, features, labels):
        self.features = torch.tensor(
            features, dtype=torch.float32
        )

        self.labels = torch.tensor(
            labels, dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_csv_dataset(file_path):
    """
    Load an intrusion detection dataset from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    dataframe : pandas.DataFrame
        Loaded dataset.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    dataframe = pd.read_csv(file_path)

    print(
        f"Dataset loaded: {file_path} "
        f"({len(dataframe)} samples)"
    )

    return dataframe


def prepare_features_and_labels(
    dataframe,
    label_column="Label"
):
    """
    Separate features and labels.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataframe.

    label_column : str
        Name of the target label column.

    Returns
    -------
    X : numpy.ndarray
        Feature matrix.

    y : numpy.ndarray
        Encoded labels.

    label_mapping : dict
        Mapping between original labels and integer labels.
    """

    if label_column not in dataframe.columns:
        raise ValueError(
            f"Label column '{label_column}' "
            f"not found in dataset."
        )

    dataframe = dataframe.dropna()

    labels = dataframe[label_column]

    features = dataframe.drop(
        columns=[label_column]
    )

    # Keep numerical features only.
    features = features.select_dtypes(
        include=[np.number]
    )

    # Encode labels.
    unique_labels = sorted(
        labels.astype(str).unique()
    )

    label_mapping = {
        label: index
        for index, label in enumerate(unique_labels)
    }

    encoded_labels = labels.astype(str).map(
        label_mapping
    ).values

    X = features.values.astype(np.float32)
    y = encoded_labels.astype(np.int64)

    return X, y, label_mapping


def create_label_heterogeneous_clients(
    X,
    y,
    num_clients,
    seed=42
):
    """
    Create clients with heterogeneous label spaces.

    Each client receives a subset of the global attack
    categories while preserving multiple samples per class.

    This function provides the basic partitioning mechanism
    used by Attack-Fed.
    """

    rng = np.random.default_rng(seed)

    unique_labels = np.unique(y)

    if num_clients > len(unique_labels):
        print(
            "Warning: number of clients is greater than "
            "the number of labels."
        )

    # Shuffle samples within each class.
    label_indices = {}

    for label in unique_labels:
        indices = np.where(y == label)[0]
        indices = rng.permutation(indices)
        label_indices[label] = indices

    clients = {
        client_id: []
        for client_id in range(num_clients)
    }

    # Assign labels to clients.
    for label in unique_labels:

        indices = label_indices[label]

        selected_clients = rng.choice(
            num_clients,
            size=min(num_clients, len(indices)),
            replace=False
        )

        splits = np.array_split(
            indices,
            len(selected_clients)
        )

        for client_id, client_indices in zip(
            selected_clients,
            splits
        ):
            clients[client_id].extend(
                client_indices.tolist()
            )

    # Build client datasets.
    client_datasets = {}

    for client_id in range(num_clients):

        indices = clients[client_id]

        if len(indices) == 0:
            continue

        indices = np.array(indices)

        client_X = X[indices]
        client_y = y[indices]

        client_datasets[client_id] = (
            client_X,
            client_y
        )

    return client_datasets


def get_client_label_spaces(client_datasets):
    """
    Extract the label space of every client.

    Returns
    -------
    dict
        Client ID -> list of local labels.
    """

    label_spaces = {}

    for client_id, (_, labels) in client_datasets.items():

        label_spaces[client_id] = sorted(
            np.unique(labels).tolist()
        )

    return label_spaces


def print_client_statistics(client_datasets):
    """
    Display basic statistics for every client.
    """

    print("\nClient statistics")
    print("-" * 60)

    for client_id, (_, labels) in client_datasets.items():

        unique_labels, counts = np.unique(
            labels,
            return_counts=True
        )

        distribution = dict(
            zip(
                unique_labels.tolist(),
                counts.tolist()
            )
        )

        print(
            f"Client {client_id:02d} | "
            f"Samples: {len(labels):6d} | "
            f"Labels: {distribution}"
        )
