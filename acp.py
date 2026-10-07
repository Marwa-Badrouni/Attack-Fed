import numpy as np


def compute_presence(labels, all_labels):
    """
    Compute the presence component of the Attack Category Profile.

    Presence indicates whether an attack category is observed
    by a client.

    Returns
    -------
    dict
        label -> 0/1
    """

    client_labels = set(np.unique(labels))

    return {
        label: int(label in client_labels)
        for label in all_labels
    }


def compute_frequency(labels, all_labels):
    """
    Compute the frequency component of the Attack Category Profile.

    Frequency represents the proportion of local samples
    belonging to each attack category.
    """

    total_samples = len(labels)

    if total_samples == 0:
        return {
            label: 0.0
            for label in all_labels
        }

    unique_labels, counts = np.unique(
        labels,
        return_counts=True
    )

    count_map = dict(
        zip(unique_labels, counts)
    )

    return {
        label: count_map.get(label, 0) / total_samples
        for label in all_labels
    }


def compute_diversity(labels, all_labels):
    """
    Compute the diversity component of the Attack Category Profile.

    Diversity captures the relative richness of attack
    categories observed by a client.

    Here, the component is normalized by the total number
    of globally available categories.
    """

    number_of_local_categories = len(
        np.unique(labels)
    )

    number_of_global_categories = len(
        all_labels
    )

    if number_of_global_categories == 0:
        return 0.0

    return (
        number_of_local_categories
        / number_of_global_categories
    )


def normalize_vector(vector):
    """
    Min-max normalization of a vector.

    If all values are identical, a zero vector is returned.
    """

    vector = np.asarray(
        vector,
        dtype=np.float64
    )

    minimum = np.min(vector)
    maximum = np.max(vector)

    if maximum == minimum:
        return np.zeros_like(vector)

    return (
        (vector - minimum)
        / (maximum - minimum)
    )


def build_acp(
    labels,
    all_labels,
    normalize=True
):
    """
    Build the Attack Category Profile (ACP).

    ACP consists of:
        - Presence
        - Frequency
        - Diversity

    Parameters
    ----------
    labels : array-like
        Local labels of a client.

    all_labels : array-like
        Global set of attack categories.

    normalize : bool
        Whether to normalize ACP components.

    Returns
    -------
    dict
        ACP representation.
    """

    presence = compute_presence(
        labels,
        all_labels
    )

    frequency = compute_frequency(
        labels,
        all_labels
    )

    diversity = compute_diversity(
        labels,
        all_labels
    )

    presence_vector = np.array(
        [presence[label] for label in all_labels],
        dtype=np.float64
    )

    frequency_vector = np.array(
        [frequency[label] for label in all_labels],
        dtype=np.float64
    )

    if normalize:

        presence_vector = normalize_vector(
            presence_vector
        )

        frequency_vector = normalize_vector(
            frequency_vector
        )

        diversity_vector = np.array(
            [diversity],
            dtype=np.float64
        )

    else:

        diversity_vector = np.array(
            [diversity],
            dtype=np.float64
        )

    return {
        "presence": presence_vector,
        "frequency": frequency_vector,
        "diversity": diversity_vector
    }


def flatten_acp(acp):
    """
    Convert an ACP dictionary into a single vector.

    This vector can directly be used by a clustering
    algorithm.
    """

    return np.concatenate([
        acp["presence"],
        acp["frequency"],
        acp["diversity"]
    ])
