from abc import ABC, abstractmethod
import torch
from torch.utils.data import TensorDataset


class BaseDataset(TensorDataset, ABC):
    """Base class for all datasets."""

    def __init__(self, split="train", subsample=0):
        self.name = split
        self.data, self.labels = self._load_data(split)

        # Subsample data if needed
        if subsample > 0:
            self.data = self.data[::subsample]
            self.labels = self.labels[::subsample]

        # Convert to torch tensors
        self.data = torch.tensor(self.data, dtype=torch.int64)
        self.labels = torch.tensor(self.labels, dtype=torch.int64)

        super().__init__(self.data, self.labels)

    @abstractmethod
    def _load_data(self, split):
        """Load data for the given split."""
        pass

    @abstractmethod
    def get_input_shape(self):
        """Get the input shape for the model."""
        pass

    @abstractmethod
    def plot(self, dataset_list, label_list, prefix="dataset", suffix=""):
        """Plot the dataset."""
        pass
