import torch
import torch.nn as nn


class LocalIDSModel(nn.Module):
    """
    Local neural network used by each federated client.

    The model produces:
        1. A latent representation
        2. Class logits

    The latent representation is later used to construct
    class prototypes for Attack-Fed knowledge transfer.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        latent_dim=32
    ):
        super(LocalIDSModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, latent_dim)
        )

        self.classifier = nn.Linear(
            latent_dim,
            num_classes
        )

    def forward(self, x):
        """
        Forward pass.

        Returns
        -------
        logits : torch.Tensor
            Classification outputs.

        representation : torch.Tensor
            Latent representation used for prototypes.
        """

        representation = self.encoder(x)

        logits = self.classifier(
            representation
        )

        return logits, representation


def build_model(
    input_dim,
    num_classes,
    latent_dim=32,
    device="cpu"
):
    """
    Build and initialize a local IDS model.
    """

    model = LocalIDSModel(
        input_dim=input_dim,
        num_classes=num_classes,
        latent_dim=latent_dim
    )

    model = model.to(device)

    return model
