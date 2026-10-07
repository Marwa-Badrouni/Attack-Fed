import numpy as np


def compute_cluster_profile(
    client_ids,
    acp_vectors
):
    """
    Compute the representative ACP profile of a cluster.

    The cluster profile is obtained by averaging the ACP
    vectors of its clients.
    """

    if len(client_ids) == 0:
        return None

    vectors = np.asarray(
        [
            acp_vectors[client_id]
            for client_id in client_ids
        ],
        dtype=np.float64
    )

    return np.mean(
        vectors,
        axis=0
    )


def cosine_similarity(vector_a, vector_b):
    """
    Compute cosine similarity between two vectors.
    """

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(
        np.dot(vector_a, vector_b)
        / (norm_a * norm_b)
    )


def build_cluster_profiles(
    clusters,
    acp_vectors
):
    """
    Build one representative ACP profile per cluster.

    Returns
    -------
    profiles : dict
        Cluster ID -> representative ACP vector.
    """

    profiles = {}

    for cluster_id, client_ids in clusters.items():

        profile = compute_cluster_profile(
            client_ids,
            acp_vectors
        )

        if profile is not None:
            profiles[cluster_id] = profile

    return profiles


def build_crg(
    clusters,
    acp_vectors,
    threshold=0.5
):
    """
    Construct the Cluster Relationship Graph (CRG).

    Nodes:
        clusters

    Edges:
        similarity between cluster profiles

    An edge is retained when the similarity is greater
    than or equal to the specified threshold.

    Returns
    -------
    graph : dict
        Cluster relationships.

    similarity_matrix : dict
        Pairwise cluster similarities.

    cluster_profiles : dict
        Representative ACP profile of each cluster.
    """

    cluster_profiles = build_cluster_profiles(
        clusters,
        acp_vectors
    )

    cluster_ids = sorted(
        cluster_profiles.keys()
    )

    graph = {
        cluster_id: []
        for cluster_id in cluster_ids
    }

    similarity_matrix = {}

    for i, cluster_i in enumerate(cluster_ids):

        similarity_matrix[cluster_i] = {}

        for j, cluster_j in enumerate(cluster_ids):

            if i == j:
                continue

            similarity = cosine_similarity(
                cluster_profiles[cluster_i],
                cluster_profiles[cluster_j]
            )

            similarity_matrix[
                cluster_i
            ][cluster_j] = similarity

            if similarity >= threshold:

                graph[cluster_i].append(
                    {
                        "target": cluster_j,
                        "similarity": similarity
                    }
                )

    return (
        graph,
        similarity_matrix,
        cluster_profiles
    )


def print_crg(graph):
    """
    Display Cluster Relationship Graph.
    """

    print("\nCluster Relationship Graph")
    print("-" * 60)

    for cluster_id, neighbors in graph.items():

        if len(neighbors) == 0:

            print(
                f"Cluster {cluster_id}: no "
                f"eligible transfer relationship"
            )

            continue

        print(
            f"Cluster {cluster_id}:"
        )

        for relation in neighbors:

            target = relation["target"]
            similarity = relation["similarity"]

            print(
                f"  -> Cluster {target} "
                f"(similarity={similarity:.4f})"
            )
