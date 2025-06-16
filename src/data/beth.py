import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .base import BaseDataset


class BETHDataset(BaseDataset):
    """BETH dataset for anomaly detection."""

    def __init__(self, split="train", subsample=1):
        """Initialize BETH dataset.

        Args:
            split (str): One of 'train', 'val', or 'test'
            subsample (int): Factor by which to subsample the data (1 means no subsampling)
        """
        self.split = split
        self.subsample = subsample
        super().__init__(split=split, subsample=subsample)

    def _load_data(self, split):
        """Load data for the given split."""
        # Load data
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
        )
        if split == "train":
            file_path = os.path.join(data_dir, "labelled_training_data.csv")
        elif split == "val":
            file_path = os.path.join(data_dir, "labelled_validation_data.csv")
        elif split == "test":
            file_path = os.path.join(data_dir, "labelled_testing_data.csv")
        else:
            raise ValueError(f"Invalid split: {split}")

        # Read data
        df = pd.read_csv(file_path)

        # Select and process specific columns
        data = pd.DataFrame(
            df[
                [
                    "processId",
                    "parentProcessId",
                    "userId",
                    "mountNamespace",
                    "eventId",
                    "argsNum",
                    "returnValue",
                ]
            ]
        )
        data["processId"] = data["processId"].map(
            lambda x: 0 if x in [0, 1, 2] else 1
        )  # Map to OS/not OS
        data["parentProcessId"] = data["parentProcessId"].map(
            lambda x: 0 if x in [0, 1, 2] else 1
        )  # Map to OS/not OS
        data["userId"] = data["userId"].map(
            lambda x: 0 if x < 1000 else 1
        )  # Map to OS/not OS
        data["mountNamespace"] = data["mountNamespace"].map(
            lambda x: 0 if x == 4026531840 else 1
        )  # Map to mount access to mnt/ (all non-OS users) /elsewhere
        data["eventId"] = data[
            "eventId"
        ]  # Keep eventId values (requires knowing max value)
        data["returnValue"] = data["returnValue"].map(
            lambda x: 0 if x == 0 else (1 if x > 0 else 2)
        )  # Map to success/success with value/error

        # Convert to numpy arrays with appropriate dtypes
        data = data.to_numpy().astype(np.int64)
        labels = df["sus"].to_numpy().astype(np.int64)

        print(f"Loaded {split} data with shape {data.shape}")
        return data, labels

    def get_input_shape(self):
        """Get the input shape for the model."""
        return self.data.shape[1]

    def plot(self, dataset_list, label_list, prefix="dataset", suffix=""):
        """Plot the dataset using PCA for dimensionality reduction."""
        from sklearn.decomposition import PCA

        # Reduce dimensionality to 2D for visualization
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(self.data)

        # Create plot
        plt.figure(figsize=(10, 8))
        plt.scatter(
            data_2d[self.labels == 0, 0],
            data_2d[self.labels == 0, 1],
            c="blue",
            label="Normal",
            alpha=0.5,
        )
        plt.scatter(
            data_2d[self.labels == 1, 0],
            data_2d[self.labels == 1, 1],
            c="red",
            label="Anomaly",
            alpha=0.5,
        )

        plt.title(f"BETH Dataset - {self.split} Split")
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.legend()

        # Save plot
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{prefix}_{self.split}_{suffix}.png")
        plt.close()
