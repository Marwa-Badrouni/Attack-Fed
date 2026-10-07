

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.prototype import compute_local_prototypes


class FederatedClient:
    """
    Federated client for Attack-Fed.

    Each client owns a local dataset with a potentially
    different label space.
    """

    def __init__(
        self,
        client_id,
        model,
        dataset,
        batch_size=64,
        learning_rate=0.001,
        local_epochs=1,
        device="cpu"
    ):

        self.client_id = client_id

        self.model = model.to(device)

        self.dataset = dataset

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs

        self.device = device

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

    

    def train(self):
        """
        Perform local training on the client's dataset.

        Returns
        -------
        training_loss : float
            Average local training loss.
        """

        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for _ in range(self.local_epochs):

            for features, labels in self.dataloader:

                features = features.to(
                    self.device
                )

                labels = labels.to(
                    self.device
                )

                self.optimizer.zero_grad()

                logits, _ = self.model(
                    features
                )

                loss = self.criterion(
                    logits,
                    labels
                )

                loss.backward()

                self.optimizer.step()

                batch_size = labels.size(0)

                total_loss += (
                    loss.item() * batch_size
                )

                total_samples += batch_size

        if total_samples == 0:
            return 0.0

        return total_loss / total_samples

   

    def get_model_parameters(self):
        """
        Return a copy of the local model parameters.
        """

        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def set_model_parameters(self, parameters):
        """
        Update the local model using global parameters.
        """

        self.model.load_state_dict(
            copy.deepcopy(parameters)
        )

        self.model.to(self.device)

    

    def compute_prototypes(self):
        """
        Compute local class prototypes.
        """

        prototypes = compute_local_prototypes(
            model=self.model,
            dataloader=self.dataloader,
            device=self.device
        )

        return prototypes

    # ========================================================
    # Local Label Space
    # ========================================================

    def get_label_space(self):
        """
        Return the set of labels available locally.
        """

        labels = set()

        for _, batch_labels in self.dataloader:

            labels.update(
                batch_labels.tolist()
            )

        return sorted(labels)

    

    def get_client_info(self):
        """
        Return basic information about the client.
        """

        return {
            "client_id": self.client_id,
            "num_samples": len(self.dataset),
            "label_space": self.get_label_space()
        }
