"""Contains a model which can make predictions on the nodes of a DAG."""
import torch
import torch.nn as nn

from .dag import RootedDAG
from chemprop.nn_utils import get_activation_function


class DAGModel(nn.Module):
    def __init__(self,
                 dag: RootedDAG,
                 hidden_size: int,
                 embedding_size: int,
                 activation: str = 'ReLU'):
        super(DAGModel, self).__init__()
        self.dag = dag
        self.node_embeddings = nn.Embedding(
            num_embeddings=self.dag.number_of_nodes(),
            embedding_dim=embedding_size
        )
        self.layer1 = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.activation = get_activation_function(activation)
        self.layer2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, embedding: torch.FloatTensor) -> torch.FloatTensor:
        # Embedding has shape (batch_size, hidden_size)

        for depth in range(self.dag.max_depth):
            pass

        return embedding
