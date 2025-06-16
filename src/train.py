import os
import pickle
import wandb
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from src.config.config import configure
from src.data.beth import BETHDataset
from src.models.ifor import IFORModel
from src.models.gnn import GNNModel

DATASETS = {"beth": BETHDataset}

MODELS = {"ifor": IFORModel, "gnn": GNNModel}


def get_model_path(model_name, dataset_name):
    """Get the path for saving/loading models."""
    os.makedirs("models", exist_ok=True)
    return os.path.join("models", f"{model_name}_{dataset_name}_best.pkl")


def train(args):
    """Train the model."""
    print("Loading data...")
    train_dataset = DATASETS[args.dataset](split="train", subsample=args.subsample)
    val_dataset = DATASETS[args.dataset](split="val", subsample=args.subsample)

    # Initialize model
    if args.model == "ifor":
        model = MODELS[args.model](
            contamination=args.outliers_fraction, random_state=args.seed
        )
    else:  # GNN model
        model = MODELS[args.model](
            input_dim=train_dataset.get_input_shape(),  # Already returns number of features
            hidden_dim=args.hidden_size,
            num_layers=args.num_layers,
            outliers_fraction=args.outliers_fraction,
            lr=args.learning_rate,
            epochs=args.gnn_epochs,
            k=args.k,
        )

    # Train model
    print("Training model...")
    train_metrics = []
    val_metrics = []
    best_val_score = float("-inf")
    patience_counter = 0

    if args.model == "ifor":
        # IFOR is trained in one go
        train_loss, model = model.fit(train_dataset.data)
        train_metrics.append(train_loss)
        val_auroc, val_acc, val_prec, val_rec, val_f1 = model.evaluate(
            val_dataset.data, val_dataset.labels
        )
        val_metrics.append(val_auroc)

        # Save model
        model_path = get_model_path(args.model, args.dataset)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        print(
            f"Train loss: {train_loss:.4f}, Val AUROC: {val_auroc:.4f}, Val Accuracy: {val_acc:.4f}, "
            f"Val Precision: {val_prec:.4f}, Val Recall: {val_rec:.4f}, Val F1: {val_f1:.4f}"
        )

        # Log metrics
        if args.use_wandb:
            wandb.log(
                {
                    "train/loss": train_loss,
                    "val/auroc": val_auroc,
                    "val/accuracy": val_acc,
                    "val/precision": val_prec,
                    "val/recall": val_rec,
                    "val/f1": val_f1,
                }
            )
    else:
        # GNN is trained epoch by epoch
        for epoch in tqdm(range(args.epochs)):
            # Train
            train_score = model.train_epoch(train_dataset)
            train_metrics.append(train_score)

            # Validate
            val_auroc, val_acc, val_prec, val_rec, val_f1 = model.evaluate(
                val_dataset.data, val_dataset.labels
            )
            val_metrics.append(val_auroc)

            print(
                f"Epoch {epoch}: Train score: {train_score:.4f}, Val AUROC: {val_auroc:.4f}, "
                f"Val Accuracy: {val_acc:.4f}, Val Precision: {val_prec:.4f}, "
                f"Val Recall: {val_rec:.4f}, Val F1: {val_f1:.4f}"
            )

            # Log metrics
            if args.use_wandb:
                wandb.log(
                    {
                        "train/score": train_score,
                        "val/auroc": val_auroc,
                        "val/accuracy": val_acc,
                        "val/precision": val_prec,
                        "val/recall": val_rec,
                        "val/f1": val_f1,
                        "epoch": epoch,
                    }
                )

            # Early stopping
            if val_auroc > best_val_score:
                best_val_score = val_auroc
                patience_counter = 0
                # Save best model
                model_path = get_model_path(args.model, args.dataset)
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break


def test(args):
    """Test the model."""
    print("Loading test data...")
    test_dataset = DATASETS[args.dataset](split="test", subsample=args.subsample)

    # Load best model
    model_path = get_model_path(args.model, args.dataset)
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except ModuleNotFoundError:
        # Handle models saved with old structure
        import sys

        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    # Evaluate
    print("Evaluating model...")
    test_auroc, test_acc, test_prec, test_rec, test_f1 = model.evaluate(
        test_dataset.data, test_dataset.labels
    )

    print(
        f"Test AUROC: {test_auroc:.4f}, Test Accuracy: {test_acc:.4f}, "
        f"Test Precision: {test_prec:.4f}, Test Recall: {test_rec:.4f}, Test F1: {test_f1:.4f}"
    )

    # Log results
    if args.use_wandb:
        wandb.log(
            {
                "test/auroc": test_auroc,
                "test/accuracy": test_acc,
                "test/precision": test_prec,
                "test/recall": test_rec,
                "test/f1": test_f1,
            }
        )


def main():
    args = configure()

    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"{args.model}_{args.dataset}",
            config=vars(args),
        )

    if args.train:
        train(args)
    if args.test:
        test(args)

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
