import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Attack-Fed: Attack-aware Federated Learning "
                    "via Prototype-guided Cross-cluster Knowledge Transfer"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="CIC-IoT-2023",
        choices=[
            "CIC-IoT-2023",
            "UNSW-NB15",
            "CIC-IDS2017",
            "CIC-IDS2018",
            "CIC-DDoS2019"
        ],
        help="Dataset used for the federated learning experiment."
    )

    parser.add_argument(
        "--num_clients",
        type=int,
        default=20,
        help="Number of federated clients."
    )

    parser.add_argument(
        "--num_rounds",
        type=int,
        default=100,
        help="Number of federated communication rounds."
    )

    parser.add_argument(
        "--local_epochs",
        type=int,
        default=1,
        help="Number of local training epochs per round."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Local training batch size."
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate used by the local optimizer."
    )

    parser.add_argument(
        "--latent_dim",
        type=int,
        default=32,
        help="Dimension of the latent representation."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 70)
    print("Attack-Fed")
    print("Attack-aware Federated Learning via Prototype-guided")
    print("Cross-cluster Knowledge Transfer")
    print("=" * 70)

    print(f"Dataset       : {args.dataset}")
    print(f"Clients       : {args.num_clients}")
    print(f"Rounds        : {args.num_rounds}")
    print(f"Local epochs  : {args.local_epochs}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Learning rate : {args.learning_rate}")
    print(f"Latent dim    : {args.latent_dim}")
    print(f"Random seed   : {args.seed}")

   


if __name__ == "__main__":
    main()
