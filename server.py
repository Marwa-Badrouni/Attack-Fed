

import torch

from server.acp import (
    build_acp,
    flatten_acp
)

from server.clustering import (
    build_client_clusters
)

from server.crg import (
    build_crg
)

from server.knowledge_transfer import (
    collect_cluster_prototypes,
    transfer_cluster_knowledge
)


class FederatedServer:
    """
    Central server for the Attack-Fed framework.

    The server coordinates:
        1. Client model aggregation
        2. Attack Category Profile construction
        3. Client clustering
        4. Cluster Relationship Graph construction
        5. Prototype aggregation
        6. Cross-cluster knowledge transfer
    """

    def __init__(
        self,
        global_model,
        num_clients,
        clustering_config=None,
        crg_config=None,
        transfer_config=None
    ):

        self.global_model = global_model

        self.num_clients = num_clients

        self.clustering_config = (
            clustering_config or {}
        )

        self.crg_config = (
            crg_config or {}
        )

        self.transfer_config = (
            transfer_config or {}
        )

        self.clients = {}

        self.client_prototypes = {}

        self.acp_vectors = {}

        self.clusters = {}

        self.crg = {}

        self.cluster_prototypes = {}

        self.transferred_knowledge = {}

    # ========================================================
    # Client Registration
    # ========================================================

    def register_client(self, client):
        """
        Register a federated client.
        """

        self.clients[
            client.client_id
        ] = client

    # ========================================================
    # Global Model
    # ========================================================

    def get_global_parameters(self):
        """
        Return a copy of the global model parameters.
        """

        return {
            key: value.detach().cpu().clone()
            for key, value
            in self.global_model.state_dict().items()
        }

    def set_global_parameters(self, parameters):
        """
        Update the global model.
        """

        self.global_model.load_state_dict(
            copy.deepcopy(parameters)
        )

    # ========================================================
    # FedAvg Aggregation
    # ========================================================

    def aggregate_models(
        self,
        client_parameters
    ):
        """
        Perform sample-weighted federated aggregation.

        Parameters
        ----------
        client_parameters : dict
            Client ID -> model parameters.

        Returns
        -------
        dict
            Aggregated global model parameters.
        """

        if not client_parameters:
            return self.get_global_parameters()

        total_samples = sum(
            len(
                self.clients[client_id].dataset
            )
            for client_id in client_parameters
        )

        if total_samples == 0:
            return self.get_global_parameters()

        global_parameters = {}

        first_client = next(
            iter(client_parameters)
        )

        for parameter_name in (
            client_parameters[first_client]
        ):

            global_parameters[
                parameter_name
            ] = torch.zeros_like(
                client_parameters[
                    first_client
                ][parameter_name]
            )

        for client_id, parameters in (
            client_parameters.items()
        ):

            weight = (
                len(
                    self.clients[client_id].dataset
                )
                / total_samples
            )

            for parameter_name, parameter in (
                parameters.items()
            ):

                global_parameters[
                    parameter_name
                ] += (
                    weight * parameter
                )

        return global_parameters

    # ========================================================
    # Attack Category Profiles
    # ========================================================

    def build_acp_profiles(self):
        """
        Construct an ACP vector for every registered client.
        """

        all_labels = set()

        for client in self.clients.values():

            all_labels.update(
                client.get_label_space()
            )

        all_labels = sorted(
            all_labels
        )

        self.acp_vectors = {}

        for client_id, client in (
            self.clients.items()
        ):

            labels = []

            for _, batch_labels in (
                client.dataloader
            ):

                labels.extend(
                    batch_labels.tolist()
                )

            acp = build_acp(
                labels=labels,
                all_labels=all_labels,
                normalize=True
            )

            self.acp_vectors[
                client_id
            ] = flatten_acp(acp)

        return self.acp_vectors

    # ========================================================
    # Client Clustering
    # ========================================================

    def cluster_clients(self):
        """
        Group clients using their ACP profiles.
        """

        if not self.acp_vectors:
            self.build_acp_profiles()

        self.clusters, _, exemplars = (
            build_client_clusters(
                self.acp_vectors,
                damping=self.clustering_config.get(
                    "damping",
                    0.9
                ),
                max_iter=self.clustering_config.get(
                    "max_iter",
                    200
                ),
                convergence_iter=self.clustering_config.get(
                    "convergence_iter",
                    15
                )
            )
        )

        return self.clusters, exemplars

    # ========================================================
    # Local Prototypes
    # ========================================================

    def collect_prototypes(self):
        """
        Collect local prototypes from all clients.
        """

        self.client_prototypes = {}

        for client_id, client in (
            self.clients.items()
        ):

            self.client_prototypes[
                client_id
            ] = client.compute_prototypes()

        return self.client_prototypes

    # ========================================================
    # Cluster Relationship Graph
    # ========================================================

    def build_cluster_relationship_graph(self):
        """
        Construct the CRG using cluster-level ACP profiles.
        """

        if not self.clusters:
            self.cluster_clients()

        self.crg, _, cluster_profiles = (
            build_crg(
                clusters=self.clusters,
                acp_vectors=self.acp_vectors,
                threshold=self.crg_config.get(
                    "threshold",
                    0.5
                )
            )
        )

        return self.crg, cluster_profiles

    # ========================================================
    # Knowledge Transfer
    # ========================================================

    def perform_knowledge_transfer(self):
        """
        Perform prototype-based cross-cluster knowledge transfer.
        """

        if not self.client_prototypes:
            self.collect_prototypes()

        if not self.clusters:
            self.cluster_clients()

        if not self.crg:
            self.build_cluster_relationship_graph()

        self.cluster_prototypes = (
            collect_cluster_prototypes(
                clusters=self.clusters,
                client_prototypes=self.client_prototypes
            )
        )

        self.transferred_knowledge = (
            transfer_cluster_knowledge(
                clusters=self.clusters,
                crg=self.crg,
                cluster_prototypes=self.cluster_prototypes,
                lambda_transfer=self.transfer_config.get(
                    "lambda_transfer",
                    0.1
                )
            )
        )

        return self.transferred_knowledge

    # ========================================================
    # One Attack-Fed Round
    # ========================================================

    def run_round(self):
        """
        Execute one federated communication round.

        Current pipeline:

            Global model
                 ↓
            Local training
                 ↓
            Model aggregation
                 ↓
            ACP construction
                 ↓
            Client clustering
                 ↓
            Prototype extraction
                 ↓
            CRG construction
                 ↓
            Knowledge transfer
        """

        # ----------------------------------------------------
        # 1. Send global model to clients
        # ----------------------------------------------------

        global_parameters = (
            self.get_global_parameters()
        )

        for client in self.clients.values():

            client.set_model_parameters(
                global_parameters
            )

        # ----------------------------------------------------
        # 2. Local training
        # ----------------------------------------------------

        client_parameters = {}

        local_losses = {}

        for client_id, client in (
            self.clients.items()
        ):

            loss = client.train()

            local_losses[
                client_id
            ] = loss

            client_parameters[
                client_id
            ] = client.get_model_parameters()

        # ----------------------------------------------------
        # 3. Global model aggregation
        # ----------------------------------------------------

        aggregated_parameters = (
            self.aggregate_models(
                client_parameters
            )
        )

        self.set_global_parameters(
            aggregated_parameters
        )

        # ----------------------------------------------------
        # 4. Build ACP
        # ----------------------------------------------------

        self.build_acp_profiles()

        # ----------------------------------------------------
        # 5. Cluster clients
        # ----------------------------------------------------

        self.cluster_clients()

        # ----------------------------------------------------
        # 6. Collect prototypes
        # ----------------------------------------------------

        self.collect_prototypes()

        # ----------------------------------------------------
        # 7. Build CRG
        # ----------------------------------------------------

        self.build_cluster_relationship_graph()

       

        self.perform_knowledge_transfer()

        return {
            "losses": local_losses,
            "clusters": self.clusters,
            "crg": self.crg,
            "prototypes": self.client_prototypes,
            "transferred_knowledge":
                self.transferred_knowledge
        }
