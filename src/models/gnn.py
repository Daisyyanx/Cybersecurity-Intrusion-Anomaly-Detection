import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.utils import to_undirected, to_dense_adj, dense_to_sparse
from torch_geometric.nn.pool import knn_graph
from torch_geometric.data import Data
import numpy as np
from tqdm.auto import tqdm
import wandb
from .base import BaseModel


class GAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        # first layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        # remaining hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        # final embedding layer
        self.convs.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        # no activation on last layer
        return self.convs[-1](x, edge_index)


class GNNModel(BaseModel):
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_layers=3,
        outliers_fraction=0.1,
        lr=1e-3,
        epochs=100,
        k=5,
    ):
        super().__init__()
        # build a PyG GAE
        self.encoder = GAEEncoder(input_dim, hidden_dim, num_layers)
        self.model = GAE(self.encoder)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.outliers_fraction = outliers_fraction
        self.lr = lr
        self.epochs = epochs
        self.k = k
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)

    def _create_graph(self, data):
        """Convert data points to a graph structure using k-nearest neighbors.
        Args:
            data: Input data as numpy array or torch.Tensor of shape [N, F]
        Returns:
            Data object with x and edge_index
        """
        # Convert to numpy if tensor
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        # Convert to tensor and move to device
        x = torch.tensor(data, dtype=torch.float, device=self.device)

        # Create k-nearest neighbors graph
        k = min(self.k, len(data) - 1)  # Use k or less if dataset is small
        edge_index = knn_graph(x, k=k, loop=False)
        edge_index = to_undirected(edge_index)  # Ensure undirected edges

        return Data(x=x, edge_index=edge_index)

    def train_epoch(self, dataset):
        """Train the model for one epoch.

        Args:
            dataset: Dataset containing the training data

        Returns:
            float: Training loss for this epoch
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Create graph for the dataset
        data = self._create_graph(dataset.data)

        # Forward pass
        z = self.model.encode(data.x, data.edge_index)
        loss = self.model.recon_loss(z, data.edge_index)

        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def fit(self, X, y=None):
        """Train the GNN model.
        Args:
            X: Input data tensor (dataset.data)
            y: Optional labels (not used in GNN training)
        """
        self.model.train()

        # Create graph for the whole dataset
        data = self._create_graph(X)

        # Training loop
        print(f"Training GNN for {self.epochs} epochs")
        for epoch in tqdm(range(self.epochs), desc="Training GNN"):
            self.optimizer.zero_grad()
            # encode → z: [n_samples, hidden_dim]
            z = self.model.encode(data.x, data.edge_index)
            # loss = edge-reconstruction loss
            loss = self.model.recon_loss(z, data.edge_index)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Log to wandb if enabled
            if wandb.run is not None:
                wandb.log(
                    {
                        "gnn/epoch": epoch,
                        "gnn/recon_loss": loss.item(),
                        "gnn/learning_rate": self.scheduler.get_last_lr()[0],
                    }
                )

        return self

    def decision_function(self, X):
        """
        Returns one score per node: its MSE between
        reconstructed adjacency row and true adjacency row.
        Higher scores indicate anomalies.
        """
        self.model.eval()
        with torch.no_grad():
            data = self._create_graph(X)
            z = self.model.encode(data.x, data.edge_index)
            # reconstruct full adjacency
            adj_rec = torch.sigmoid(z @ z.t())  # [N,N]
            adj_true = to_dense_adj(data.edge_index)[0].to(adj_rec)  # [N,N]
            # per-node MSE (higher values indicate anomalies)
            errors = ((adj_rec - adj_true) ** 2).mean(dim=1)
            # Normalize scores to [0, 1] range
            errors = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)
        return errors.cpu().numpy()

    def predict(self, X):
        """
        Label as anomaly (1) any node whose
        recon-error is above the given percentile.
        """
        scores = self.decision_function(X)
        thresh = np.percentile(scores, 100 - 100 * self.outliers_fraction)
        predictions = np.where(scores > thresh, 1, 0)

        return predictions
