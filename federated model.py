import torch


def aggregate_parameters(client_parameters, client_weights=None):
    """
    Aggregate client model parameters using Federated Averaging.

    Parameters
    ----------
    client_parameters : list of dict
        Model parameters received from clients.

    client_weights : list of float, optional
        Weight of each client, typically proportional to
        the number of local training samples.

    Returns
    -------
    dict
        Aggregated global model parameters.
    """

    if not client_parameters:
        raise ValueError("client_parameters cannot be empty.")

    if client_weights is None:
        client_weights = [
            1.0 / len(client_parameters)
            for _ in client_parameters
        ]

    if len(client_parameters) != len(client_weights):
        raise ValueError(
            "The number of client parameter sets must "
            "match the number of client weights."
        )

    # Normalize weights
    total_weight = sum(client_weights)

    if total_weight <= 0:
        raise ValueError("Client weights must sum to a positive value.")

    normalized_weights = [
        weight / total_weight
        for weight in client_weights
    ]

    # Initialize the global parameters
    global_parameters = {}

    first_parameters = client_parameters[0]

    for key in first_parameters.keys():

        global_parameters[key] = torch.zeros_like(
            first_parameters[key],
            dtype=first_parameters[key].dtype
        )

        for parameters, weight in zip(
            client_parameters,
            normalized_weights
        ):
            global_parameters[key] += (
                parameters[key] * weight
            )

    return global_parameters


def fedavg(
    global_model,
    client_parameters,
    client_weights=None
):
    """
    Perform one FedAvg aggregation step.

    Parameters
    ----------
    global_model : torch.nn.Module
        Global federated model.

    client_parameters : list of dict
        Local model parameters from clients.

    client_weights : list of float, optional
        Client aggregation weights.

    Returns
    -------
    torch.nn.Module
        Updated global model.
    """

    aggregated_parameters = aggregate_parameters(
        client_parameters,
        client_weights
    )

    global_model.load_state_dict(
        aggregated_parameters
    )

  client_parameters = [
    client_1.get_model_parameters(),
    client_2.get_model_parameters(),
    client_3.get_model_parameters()
]

client_weights = [
    len(client_1.dataset),
    len(client_2.dataset),
    len(client_3.dataset)
]

global_model = fedavg(
    global_model,
    client_parameters,
    client_weights
)

def proximal_loss(local_model, global_parameters):
    """
    Compute the FedProx proximal regularization term.

    Parameters
    ----------
    local_model : torch.nn.Module
        Current local client model.

    global_parameters : dict
        Parameters of the global model before local training.

    Returns
    -------
    torch.Tensor
        Proximal regularization term.
    """

    loss = torch.tensor(
        0.0,
        device=next(local_model.parameters()).device
    )

    for name, parameter in local_model.named_parameters():

        if name in global_parameters:
            global_parameter = global_parameters[name].to(
                parameter.device
            )

            loss += torch.sum(
                (parameter - global_parameter) ** 2
            )

    return 0.5 * loss


def fedprox_loss(
    classification_loss,
    local_model,
    global_parameters,
    mu=0.01
):
    """
    Compute the complete FedProx local objective.

    L_FedProx = L_CE + mu * L_prox

    Parameters
    ----------
    classification_loss : torch.Tensor
        Local classification loss.

    local_model : torch.nn.Module
        Current local model.

    global_parameters : dict
        Global model parameters before local training.

    mu : float
        FedProx proximal coefficient.

    Returns
    -------
    torch.Tensor
        FedProx objective.
    """

    prox_loss = proximal_loss(
        local_model,
        global_parameters
    )

    total_loss = (
        classification_loss
        + mu * prox_loss
    )

    return total_loss


def aggregate_parameters(
    client_parameters,
    client_weights=None
):
    """
    Aggregate client parameters.

    FedProx uses the same server-side aggregation
    mechanism as FedAvg.

    Parameters
    ----------
    client_parameters : list of dict
        Local model parameters.

    client_weights : list of float, optional
        Client aggregation weights.

    Returns
    -------
    dict
        Aggregated global parameters.
    """

    if not client_parameters:
        raise ValueError(
            "client_parameters cannot be empty."
        )

    if client_weights is None:
        client_weights = [
            1.0 / len(client_parameters)
            for _ in client_parameters
        ]

    if len(client_parameters) != len(client_weights):
        raise ValueError(
            "Number of parameter sets and weights "
            "must be identical."
        )

    total_weight = sum(client_weights)

    if total_weight <= 0:
        raise ValueError(
            "Client weights must sum to a positive value."
        )

    normalized_weights = [
        weight / total_weight
        for weight in client_weights
    ]

    global_parameters = {}

    for key in client_parameters[0].keys():

        global_parameters[key] = torch.zeros_like(
            client_parameters[0][key]
        )

        for parameters, weight in zip(
            client_parameters,
            normalized_weights
        ):
            global_parameters[key] += (
                parameters[key] * weight
            )

    return global_parameters


def fedprox(
    global_model,
    client_parameters,
    client_weights=None
):
    """
    Perform server-side FedProx aggregation.

    Note:
        The proximal term is applied during local training.
        Server aggregation remains weighted averaging.
    """

    aggregated_parameters = aggregate_parameters(
        client_parameters,
        client_weights
    )

    global_model.load_state_dict(
        aggregated_parameters
    )

    return global_model

  def prototype_distance(
    local_prototype,
    global_prototype
):
    """
    Compute the squared Euclidean distance
    between local and global prototypes.
    """

    return torch.sum(
        (local_prototype - global_prototype) ** 2
    )


def fedproto_loss(
    classification_loss,
    local_prototypes,
    global_prototypes,
    lambda_proto=0.1
):
    """
    Compute the FedProto local objective.

    L_FedProto =
        L_CE + lambda_proto * L_proto

    where L_proto minimizes the distance between
    local and global class prototypes.

    Parameters
    ----------
    classification_loss : torch.Tensor
        Local classification loss.

    local_prototypes : dict
        Local class prototypes.

    global_prototypes : dict
        Global class prototypes.

    lambda_proto : float
        Prototype regularization coefficient.

    Returns
    -------
    torch.Tensor
        FedProto objective.
    """

    if not local_prototypes:
        return classification_loss

    prototype_loss = torch.tensor(
        0.0,
        device=classification_loss.device
    )

    matched_classes = 0

    for label, local_prototype in local_prototypes.items():

        if label not in global_prototypes:
            continue

        global_prototype = global_prototypes[label].to(
            local_prototype.device
        )

        prototype_loss += prototype_distance(
            local_prototype,
            global_prototype
        )

        matched_classes += 1

    if matched_classes > 0:
        prototype_loss /= matched_classes

    total_loss = (
        classification_loss
        + lambda_proto * prototype_loss
    )

    return total_loss


def fedproto(
    client_prototypes,
    client_weights=None
):
    """
    Perform the server-side FedProto prototype aggregation.
    """

    return aggregate_prototypes(
        client_prototypes,
        client_weights
    )

  def initialize_cluster_models(
    global_model,
    num_clusters
):
    """
    Initialize one model for each IFCA cluster.

    All cluster models start from the same global model.
    """

    cluster_models = {}

    global_parameters = global_model.state_dict()

    for cluster_id in range(num_clusters):

        cluster_models[cluster_id] = {
            key: value.detach().clone()
            for key, value in global_parameters.items()
        }

    return cluster_models


def assign_clients_to_clusters(
    client_losses
):
    """
    Assign each client to the cluster corresponding
    to its lowest local loss.

    Parameters
    ----------
    client_losses : dict
        Dictionary with:

            client_losses[client_id][cluster_id] = loss

    Returns
    -------
    dict
        client_id -> selected cluster_id
    """

    assignments = {}

    for client_id, losses in client_losses.items():

        if not losses:
            continue

        selected_cluster = min(
            losses,
            key=losses.get
        )

        assignments[client_id] = selected_cluster

    return assignments


def aggregate_ifca_clusters(
    client_parameters,
    client_assignments,
    client_weights=None
):

    return global_model
