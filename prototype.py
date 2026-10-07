import torch


@torch.no_grad()
def compute_local_prototypes(
    model,
    dataloader,
    device="cpu"
):
    """
    Compute class-wise prototypes from a trained local model.

    A prototype is the mean latent representation of all
    samples belonging to a given class.

    Parameters
    ----------
    model : torch.nn.Module
        Trained local IDS model.

    dataloader : torch.utils.data.DataLoader
        Local client data.

    device : str
        Computation device.

    Returns
    -------
    prototypes : dict
        Mapping:
            class_id -> prototype tensor
    """

    model.eval()

    representations = {}
    counts = {}

    for features, labels in dataloader:

        features = features.to(device)
        labels = labels.to(device)

        _, latent = model(features)

        for representation, label in zip(
            latent,
            labels
        ):

            class_id = int(label.item())

            if class_id not in representations:
                representations[class_id] = (
                    representation.detach().clone()
                )

                counts[class_id] = 1

            else:
                representations[class_id] += (
                    representation.detach()
                )

                counts[class_id] += 1

    prototypes = {}

    for class_id in representations:

        prototypes[class_id] = (
            representations[class_id]
            / counts[class_id]
        )

    return prototypes


def normalize_prototypes(prototypes):
    """
    L2-normalize prototype vectors.

    This can be used before prototype similarity
    computation.
    """

    normalized = {}

    for class_id, prototype in prototypes.items():

        norm = torch.norm(
            prototype,
            p=2
        )

        if norm > 0:
            normalized[class_id] = (
                prototype / norm
            )
        else:
            normalized[class_id] = prototype

    return normalized


def prototype_similarity(
    prototype_a,
    prototype_b
):
    """
    Compute cosine similarity between two prototypes.
    """

    similarity = torch.nn.functional.cosine_similarity(
        prototype_a.unsqueeze(0),
        prototype_b.unsqueeze(0)
    )

    return float(similarity.item())
