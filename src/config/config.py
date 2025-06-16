import argparse
import torch
import numpy as np


def configure():
    """Configure command line arguments."""
    parser = argparse.ArgumentParser(description="Anomaly Detection with IFOR and GNN")

    # General flags
    parser.add_argument("--dataset", type=str, default="beth", help="Dataset to use")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--subsample", type=int, default=1, help="Subsample factor")
    parser.add_argument("--vis", action="store_true", help="Visualize results")

    # Training/Testing flags
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--test", action="store_true", help="Test model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument(
        "--patience", type=int, default=3, help="Early stopping patience"
    )

    # Model flags
    parser.add_argument("--model", type=str, default="ifor", help="Model to use")
    parser.add_argument(
        "--outliers-fraction", type=float, default=0.1, help="Outliers fraction"
    )

    # GNN-specific parameters
    parser.add_argument(
        "--gnn-epochs", type=int, default=100, help="Number of GNN epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors")

    # Wandb configuration
    parser.add_argument("--use-wandb", action="store_true", help="Use wandb")
    parser.add_argument("--wandb-entity", type=str, default="", help="Wandb entity")
    parser.add_argument(
        "--wandb-project", type=str, default="BETH", help="Wandb project"
    )
    parser.add_argument("--wandb-name", type=str, default="", help="Wandb run name")
    parser.add_argument(
        "--wandb-tags", type=str, nargs="+", default=[], help="Wandb tags"
    )

    args = parser.parse_args()

    # Set device
    args.device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args
