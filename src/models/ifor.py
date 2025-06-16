import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


class IFORModel:
    def __init__(self, contamination=0.1, random_state=42):
        """Initialize the Isolation Forest model.

        Args:
            contamination (float): The proportion of outliers in the data set.
            random_state (int): Random state for reproducibility.
        """
        self.model = IsolationForest(
            contamination=contamination, random_state=random_state, n_jobs=-1
        )
        self.contamination = contamination

    def fit(self, data):
        """Fit the model to the data.

        Args:
            data (np.ndarray): Input data of shape (n_samples, n_features)

        Returns:
            tuple: (loss, self) where loss is the negative mean anomaly score
        """
        self.model.fit(data)
        scores = -self.model.score_samples(data)  # Negative scores for consistency
        return np.mean(scores), self

    def decision_function(self, data):
        """Compute anomaly scores for each sample.

        Args:
            data (np.ndarray): Input data of shape (n_samples, n_features)

        Returns:
            np.ndarray: Anomaly scores (higher values indicate anomalies)
        """
        return -self.model.score_samples(data)  # Negative scores for consistency

    def predict(self, data):
        """Predict anomaly labels (1 for anomaly, 0 for normal).

        Args:
            data (np.ndarray): Input data of shape (n_samples, n_features)

        Returns:
            np.ndarray: Binary predictions (1 for anomaly, 0 for normal)
        """
        # IsolationForest returns -1 for anomalies and 1 for normal
        # We convert to 1 for anomalies and 0 for normal
        return (self.model.predict(data) == -1).astype(int)

    def evaluate(self, data, labels):
        """Evaluate the model on the given data.

        Args:
            data (np.ndarray): Input data of shape (n_samples, n_features)
            labels (np.ndarray or torch.Tensor): True labels (1 for anomaly, 0 for normal)

        Returns:
            tuple: (auroc, accuracy, precision, recall, f1) where:
                - auroc is the area under ROC curve
                - accuracy is the overall accuracy
                - precision is the precision score
                - recall is the recall score
                - f1 is the F1 score
        """
        # Convert data to numpy if it's a tensor
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        scores = self.decision_function(data)
        predictions = self.predict(data)

        # Calculate metrics
        auroc = roc_auc_score(labels, scores)
        accuracy = np.mean(predictions == labels)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)

        return auroc, accuracy, precision, recall, f1
