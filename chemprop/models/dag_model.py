"""Contains a model which can make predictions on the nodes of a DAG."""
import torch
import torch.nn as nn

from .dag import RootedDAG
from chemprop.nn_utils import get_activation_function


def parent_index_select(node_vecs: torch.FloatTensor,
                        parent_indices: torch.FloatTensor) -> torch.FloatTensor:
    pass


class DAGModel(nn.Module):
    def __init__(self,
                 dag: RootedDAG,
                 hidden_size: int,
                 embedding_size: int,
                 activation: str = 'ReLU'):
        super(DAGModel, self).__init__()
        self.dag = dag
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.node_embeddings = nn.Embedding(
            num_embeddings=self.dag.number_of_nodes(),
            embedding_dim=embedding_size,
            padding_idx=0
        )
        self.layer1 = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.activation = get_activation_function(activation)
        self.layer2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, embedding: torch.FloatTensor  # (batch_size, hidden_size)
                ) -> torch.FloatTensor:  # (batch_size, num_nodes)
        # Root vector is equal to input embedding
        node_vecs = embedding.unsqueeze(dim=1)  # (batch_size, 1, hidden_size)
        batch_size = node_vecs.size(0)

        # Computing
        for depth in range(1, self.dag.max_depth + 1):
            # Get nodes at this depth, sorted by index
            nodes = self.dag.depth_to_nodes(depth)
            nodes, node_indices = zip(*sorted([(node, self.dag.node_to_index(node)) for node in nodes],
                                              key=lambda node_index: node_index[1]))
            node_indices = torch.LongTensor(node_indices)
            node_embeddings = self.node_embeddings(node_indices)

            # Get parent vecs
            max_num_parents = self.dag.max_num_parents_at_depth(depth)
            parent_vecs_size = (
                batch_size,
                len(nodes),
                max_num_parents,
                self.hidden_size
            )
            parent_indices = []
            for node in nodes:
                parents = sorted(self.dag.parents(node))
                parents += [0] * (max_num_parents - len(parents))  # TODO: need to do padding on node_vecs
            parent_indices = torch.LongTensor(parent_indices)
            parent_vecs = node_vecs.index_select(dim=1, index=parent_indices.view(-1))  # (batch_size, num_nodes * max_num_parents, hidden_size)
            parent_vecs = parent_vecs.view(parent_vecs_size)  # (batch_size, num_nodes, max_num_parents, hidden_size)

            # Sum parent vecs to get a single parent vector for each node
            parent_vecs = parent_vecs.sum(dim=2)  # (batch_size, num_nodes, hidden_size)

            # Apply neural network layers, incorporating node embeddings
            # TODO: FINISH THIS

        return embedding
