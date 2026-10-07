import torch


def aggregate_prototypes(
    prototypes,
    weights=None
):
    """
    Aggregate a collection of prototypes.

    Parameters
    ----------
    prototypes : list of torch.Tensor
        Prototypes representing the same attack category.

    weights : list of float, optional
        Aggregation weights.

    Returns
    -------
    torch.Tensor
        Aggregated prototype.
    """

    if len(prototypes) == 0:
        return None

    if weights is None:
        weights = [
            1.0 / len(prototypes)
            for _ in prototypes
        ]

    aggregated = torch.zeros_like(
        prototypes[0]
    )

    total_weight = sum(weights)

    if total_weight == 0:
        return aggregated

    for prototype, weight in zip(
        prototypes,
        weights
    ):
        aggregated += (
            prototype * weight
        )

    aggregated /= total_weight

    return aggregated


def collect_cluster_prototypes(
    clusters,
    client_prototypes
):
    """
    Aggregate client prototypes inside each cluster.

    Parameters
    ----------
    clusters : dict
        Cluster ID -> client IDs.

    client_prototypes : dict
        Client ID -> class prototypes.

    Returns
    -------
    cluster_prototypes : dict
        Cluster ID -> attack class -> prototype.
    """

    cluster_prototypes = {}

    for cluster_id, client_ids in clusters.items():

        class_prototypes = {}

        for client_id in client_ids:

            if client_id not in client_prototypes:
                continue

            local_prototypes = (
                client_prototypes[client_id]
            )

            for class_id, prototype in (
                local_prototypes.items()
            ):

                if class_id not in class_prototypes:
                    class_prototypes[class_id] = []

                class_prototypes[class_id].append(
                    prototype
                )

        cluster_prototypes[cluster_id] = {}

        for class_id, prototypes in (
            class_prototypes.items()
        ):

            aggregated = aggregate_prototypes(
                prototypes
            )

            if aggregated is not None:
                cluster_prototypes[
                    cluster_id
                ][class_id] = aggregated

    return cluster_prototypes


def get_transferable_clusters(
    source_cluster,
    crg
):
    """
    Return clusters that can receive knowledge
    from the source cluster according to the CRG.
    """

    relationships = crg.get(
        source_cluster,
        []
    )

    return [
        relation["target"]
        for relation in relationships
    ]


def transfer_cluster_knowledge(
    clusters,
    crg,
    cluster_prototypes,
    lambda_transfer=0.1
):
    """
    Build prototype-based cross-cluster knowledge.

    Knowledge is transferred only across CRG edges.

    Parameters
    ----------
    clusters : dict
        Cluster assignments.

    crg : dict
        Cluster Relationship Graph.

    cluster_prototypes : dict
        Cluster-level class prototypes.

    lambda_transfer : float
        Strength of transferred knowledge.

    Returns
    -------
    transferred_knowledge : dict
        Target cluster -> attack class -> list of
        weighted source prototypes.
    """

    transferred_knowledge = {
        cluster_id: {}
        for cluster_id in clusters
    }

    for source_cluster in clusters:

        if source_cluster not in cluster_prototypes:
            continue

        source_prototypes = cluster_prototypes[
            source_cluster
        ]

        target_clusters = (
            get_transferable_clusters(
                source_cluster,
                crg
            )
        )

        for target_cluster in target_clusters:

            if target_cluster not in (
                transferred_knowledge
            ):
                continue

            for class_id, prototype in (
                source_prototypes.items()
            ):

                if class_id not in (
                    transferred_knowledge[
                        target_cluster
                    ]
                ):
                    transferred_knowledge[
                        target_cluster
                    ][class_id] = []

                transferred_knowledge[
                    target_cluster
                ][class_id].append(
                    {
                        "prototype": prototype,
                        "weight": lambda_transfer,
                        "source_cluster":
                            source_cluster
                    }
                )

    return transferred_knowledge


def merge_local_and_transferred_prototypes(
    local_prototypes,
    transferred_knowledge
):
    """
    Combine local prototypes with transferred
    prototype knowledge.

    Local knowledge is preserved, while transferred
    prototypes are incorporated for shared or missing
    attack categories.
    """

    enhanced_prototypes = {
        class_id: prototype.clone()
        for class_id, prototype
        in local_prototypes.items()
    }

    for class_id, knowledge_list in (
        transferred_knowledge.items()
    ):

        if len(knowledge_list) == 0:
            continue

        source_prototypes = [
            item["prototype"]
            for item in knowledge_list
        ]

        source_weights = [
            item["weight"]
            for item in knowledge_list
        ]

        transferred_prototype = (
            aggregate_prototypes(
                source_prototypes,
                source_weights
            )
        )

        if transferred_prototype is None:
            continue

        if class_id in enhanced_prototypes:

            
            enhanced_prototypes[class_id] = (
                0.5 * enhanced_prototypes[class_id]
                + 0.5 * transferred_prototype
            )

        else:

           
            enhanced_prototypes[class_id] = (
                transferred_prototype
            )

    return enhanced_prototypes
