"""Contains a model which can make predictions on the nodes of a DAG."""
import torch
import torch.nn as nn

from chemprop.dag import RootedDAG
from chemprop.nn_utils import get_activation_function


class BatchedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(BatchedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.Tensor(1, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.FloatTensor  # (batch_size, out_features, in_features)
                ) -> torch.FloatTensor:
        output = torch.sum(input * self.W, dim=2)  # (batch_size, out_features)

        if self.bias is not None:
            output += self.bias  # (batch_size, out_features)

        return output

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class DAGModel(nn.Module):
    def __init__(self,
                 dag: RootedDAG,
                 hidden_size: int,
                 embedding_size: int,
                 activation: str = 'ReLU'):
        super(DAGModel, self).__init__()
        self.dag = dag
        self.num_nodes = self.dag.number_of_nodes()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.node_embeddings = nn.Embedding(
            num_embeddings=self.num_nodes + 1,  # Plus 1 for padding
            embedding_dim=embedding_size,
            padding_idx=0
        )
        self.layer1 = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.activation = get_activation_function(activation)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = BatchedLinear(in_features=self.hidden_size, out_features=self.num_nodes)

    def forward(self, embedding: torch.FloatTensor  # (batch_size, hidden_size)
                ) -> torch.FloatTensor:  # (batch_size, num_nodes)
        # Get batch size and device
        batch_size = embedding.size(0)
        device = embedding.device

        # Root vector is equal to input embedding
        node_vecs = embedding.unsqueeze(dim=1)  # (batch_size, 1, hidden_size)
        padding = torch.zeros_like(node_vecs)  # (batch_size, 1, hidden_size)
        node_vecs = torch.cat((padding, node_vecs), dim=1)  # (batch_size, 1, hidden_size)

        # Computing
        for depth in range(1, self.dag.max_depth + 1):
            # Get nodes at this depth, sorted by index
            nodes = sorted(self.dag.depth_to_nodes(depth), key=lambda node: self.dag.node_to_index(node))
            num_nodes = len(nodes)
            node_indices = torch.LongTensor([self.dag.node_to_index(node) for node in nodes]).to(device)
            node_embeddings = self.node_embeddings(node_indices)  # (num_nodes, embedding_size)
            node_embeddings = node_embeddings.unsqueeze(dim=0).repeat(batch_size, 1, 1)  # (batch_size, num_nodes, embedding_size)

            # Get parent vecs
            max_num_parents = self.dag.max_num_parents_at_depth(depth)
            parent_vecs_size = (
                batch_size,
                num_nodes,
                max_num_parents,
                self.hidden_size
            )

            parent_indices = []
            for node in nodes:
                parents = sorted([self.dag.node_to_index(node) for node in self.dag.parents(node)])
                parents += [0] * (max_num_parents - len(parents))  # TODO: need to do padding on node_vecs
                parent_indices.append(parents)
            parent_indices = torch.LongTensor(parent_indices).to(device)
            parent_vecs = node_vecs.index_select(dim=1, index=parent_indices.view(-1))  # (batch_size, num_nodes * max_num_parents, hidden_size)
            parent_vecs = parent_vecs.view(parent_vecs_size)  # (batch_size, num_nodes, max_num_parents, hidden_size)

            # Sum parent vecs to get a single parent vector for each node
            parent_vecs = parent_vecs.sum(dim=2)  # (batch_size, num_nodes, hidden_size)

            # Concatenate parent vecs and node vecs
            vecs = torch.cat((parent_vecs, node_embeddings), dim=2)  # (batch_size, num_nodes, hidden_size + embedding_size)

            # Apply neural network layers
            vecs = vecs.view(-1, vecs.size(2))  # (batch_size * num_nodes, hidden_size + embedding_size)
            vecs = self.layer1(vecs)  # (batch_size * num_nodes, hidden_size)
            vecs = self.activation(vecs)  # (batch_size * num_nodes, hidden_size)
            vecs = self.layer2(vecs)  # (batch_size * num_nodes, hidden_size)
            vecs = vecs.view((batch_size, num_nodes, self.hidden_size))  # (batch_size, num_nodes, hidden_size)

            # Skip connection
            vecs = parent_vecs + vecs  # (batch_size, num_nodes, hidden_size)

            # Concatenate vecs with node vecs
            node_vecs = torch.cat((node_vecs, vecs), dim=1)  # (batch_size, cumulative_num_nodes, hidden_size)

        # Remove padding
        node_vecs = node_vecs[:, 1:, :]  # (batch_size, total_num_nodes, hidden_size)

        # Final node predictions using node vecs
        node_vecs = self.output_layer(node_vecs)

        return node_vecs
