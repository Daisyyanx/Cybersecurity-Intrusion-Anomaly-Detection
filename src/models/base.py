import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


class BaseModel:
    """Base class for all anomaly detection models."""

    def fit(self, X, y=None):
        """Fit the model to the data.

        Args:
            X: Input data
            y: Optional labels (not used in unsupervised models)

        Returns:
            self: The fitted model
        """
        raise NotImplementedError

    def predict(self, X):
        """Predict anomaly labels (1 for anomaly, 0 for normal).

        Args:
            X: Input data

        Returns:
            np.ndarray: Binary predictions
        """
        raise NotImplementedError

    def decision_function(self, X):
        """Compute anomaly scores for each sample.

        Args:
            X: Input data

        Returns:
            np.ndarray: Anomaly scores (higher values indicate anomalies)
        """
        raise NotImplementedError

    def evaluate(self, X, y):
        """Evaluate the model on the given data.

        Args:
            X: Input data
            y: True labels (1 for anomaly, 0 for normal)

        Returns:
            tuple: (auroc, accuracy, precision, recall, f1) where:
                - auroc is the area under ROC curve
                - accuracy is the overall accuracy
                - precision is the precision score
                - recall is the recall score
                - f1 is the F1 score
        """
        # Convert to numpy if tensors
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        scores = self.decision_function(X)
        predictions = self.predict(X)

        # Calculate metrics
        auroc = roc_auc_score(y, scores)
        accuracy = np.mean(predictions == y)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        f1 = f1_score(y, predictions)

        return auroc, accuracy, precision, recall, f1
