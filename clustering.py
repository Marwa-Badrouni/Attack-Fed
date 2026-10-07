import numpy as np

from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity


def build_similarity_matrix(acp_vectors):
    """
    Build a cosine similarity matrix between client ACP vectors.

    Parameters
    ----------
    acp_vectors : dict
        Client ID -> ACP vector.

    Returns
    -------
    client_ids : list
        Ordered client IDs.

    similarity_matrix : numpy.ndarray
        Pairwise similarity matrix.
    """

    client_ids = sorted(acp_vectors.keys())

    matrix = np.asarray(
        [
            acp_vectors[client_id]
            for client_id in client_ids
        ],
        dtype=np.float64
    )

    similarity_matrix = cosine_similarity(matrix)

    return client_ids, similarity_matrix


def perform_affinity_propagation(
    similarity_matrix,
    damping=0.9,
    max_iter=200,
    convergence_iter=15
):
    """
    Perform Affinity Propagation using a precomputed
    similarity matrix.

    Parameters
    ----------
    similarity_matrix : numpy.ndarray
        Pairwise client similarity matrix.

    damping : float
        Damping factor of Affinity Propagation.

    max_iter : int
        Maximum number of iterations.

    convergence_iter : int
        Number of iterations with stable exemplars required
        for convergence.

    Returns
    -------
    labels : numpy.ndarray
        Cluster assignment for every client.

    cluster_centers : numpy.ndarray
        Indices of cluster exemplars.
    """

    clustering = AffinityPropagation(
        affinity="precomputed",
        damping=damping,
        max_iter=max_iter,
        convergence_iter=convergence_iter,
        random_state=42
    )

    clustering.fit(similarity_matrix)

    return (
        clustering.labels_,
        clustering.cluster_centers_indices_
    )


def build_client_clusters(
    acp_vectors,
    damping=0.9,
    max_iter=200,
    convergence_iter=15
):
    """
    Group clients according to their ACP similarity.

    Returns
    -------
    clusters : dict
        Cluster ID -> list of client IDs.

    similarity_matrix : numpy.ndarray
        Client similarity matrix.

    exemplars : dict
        Cluster ID -> exemplar client ID.
    """

    if len(acp_vectors) == 0:
        return {}, np.empty((0, 0)), {}

    client_ids, similarity_matrix = (
        build_similarity_matrix(acp_vectors)
    )

    labels, center_indices = (
        perform_affinity_propagation(
            similarity_matrix,
            damping=damping,
            max_iter=max_iter,
            convergence_iter=convergence_iter
        )
    )

    clusters = {}

    for client_id, cluster_label in zip(
        client_ids,
        labels
    ):

        if cluster_label not in clusters:
            clusters[cluster_label] = []

        clusters[cluster_label].append(
            client_id
        )

    exemplars = {}

    for cluster_id, center_index in enumerate(
        center_indices
    ):

        if center_index < len(client_ids):
            exemplars[cluster_id] = (
                client_ids[center_index]
            )

    return (
        clusters,
        similarity_matrix,
        exemplars
    )


def print_clusters(clusters, exemplars=None):
    """
    Display clustering results.
    """

    print("\nClient clusters")
    print("-" * 60)

    for cluster_id, clients in clusters.items():

        message = (
            f"Cluster {cluster_id}: "
            f"{clients}"
        )

        if exemplars is not None:
            exemplar = exemplars.get(
                cluster_id,
                None
            )

            if exemplar is not None:
                message += (
                    f" | Exemplar: Client {exemplar}"
                )

        print(message)
