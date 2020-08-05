"""Contains a model which can make predictions on the nodes of a DAG."""
from typing import List, Tuple

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

        return output  # (batch_size, out_feature)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class MLP(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 activation: str = 'ReLU'):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(in_features, hidden_features)
        self.activation = get_activation_function(activation)
        self.layer2 = nn.Linear(hidden_features, out_features)

    def forward(self, input: torch.FloatTensor  # (batch_size, in_features)
                ) -> torch.FloatTensor:  # (batch_size, out_features)
        output = self.layer1(input)  # (batch_size, hidden_features)
        output = self.activation(output)  # (batch_size, hidden_features)
        output = self.layer2(output)  # (batch_size, out_features)

        return output  # (batch_size, out_features)


class DAGModel(nn.Module):
    def __init__(self,
                 dag: RootedDAG,
                 input_size: int,
                 hidden_size: int,
                 embedding_size: int = None,
                 layer_type: str = 'shared',
                 activation: str = 'ReLU'):
        super(DAGModel, self).__init__()

        if layer_type not in ['shared', 'per_depth', 'per_node']:
            raise ValueError(f'Layer type "{layer_type}" is not supported.')

        if layer_type in ['shared', 'per_node'] and embedding_size is None:
            raise ValueError('Embedding size must be provided if layer_type is either "shared" or "per_depth"')

        self.dag = dag
        self.num_nodes = self.dag.number_of_nodes()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.layer_type = layer_type

        self.input_mlp = MLP(
            in_features=input_size,
            hidden_features=hidden_size,
            out_features=hidden_size,
            activation=activation
        )

        if self.layer_type in ['shared', 'per_depth']:
            self.node_embeddings = nn.Embedding(
                num_embeddings=self.num_nodes + 1,  # Plus 1 for padding
                embedding_dim=embedding_size,
                padding_idx=0
            )

            self.mlps = nn.ModuleList([
                MLP(
                    in_features=hidden_size + embedding_size,
                    hidden_features=hidden_size,
                    out_features=hidden_size
                )
                for _ in range(self.dag.max_depth if self.layer_type == 'per_depth' else 1)
            ])
        elif self.layer_type == 'per_node':
            raise NotImplementedError
        else:
            raise ValueError(f'Layer type "{self.layer_type}" is not supported.')

        self.output_layer = BatchedLinear(in_features=self.hidden_size, out_features=self.num_nodes)

    @property
    def device(self) -> torch.device:
        """Get the current device of the model."""
        return next(self.parameters()).device

    def get_nodes_and_indices(self, depth: int) -> Tuple[List[str], torch.LongTensor]:  # (num_nodes,)
        """Gets nodes and indices at this depth, with both sorted according to node index."""
        nodes = sorted(self.dag.depth_to_nodes(depth), key=lambda node: self.dag.node_to_index(node))
        node_indices = torch.LongTensor([self.dag.node_to_index(node) for node in nodes]).to(self.device)

        return nodes, node_indices  # (num_nodes,)

    def get_parent_indices(self,
                           depth: int,
                           nodes: List[str]  # (num_nodes,)
                           ) -> torch.LongTensor:  # (num_nodes, max_num_parents)
        """Gets the indices of the parents with 0 as padding."""
        max_num_parents = self.dag.max_num_parents_at_depth(depth)

        parent_indices = []
        for node in nodes:
            parents = sorted([self.dag.node_to_index(node) for node in self.dag.parents(node)])
            parents += [0] * (max_num_parents - len(parents))
            parent_indices.append(parents)
        parent_indices = torch.LongTensor(parent_indices).to(self.device)  # (num_nodes, max_num_parents)

        return parent_indices  # (num_nodes, max_num_parents)

    def get_parent_vecs(self,
                        depth: int,
                        nodes: List[str],  # (num_nodes,)
                        node_vecs: torch.FloatTensor  # (batch_size, cumulative_num_nodes, hidden_size)
                        ) -> torch.FloatTensor:  # (batch_size, num_nodes, hidden_size)
        """Gets the parent vectors, summing across parents to get a single input parent vector for each node."""
        # Get parent indices
        parent_indices = self.get_parent_indices(depth=depth, nodes=nodes)  # (num_nodes, max_num_parents)

        # Determine parent vecs size
        parent_vecs_size = (
            node_vecs.size(0),       # batch_size
            parent_indices.size(0),  # num_nodes
            parent_indices.size(1),  # max_num_parents
            self.hidden_size         # hidden_size
        )

        # Get parent vecs
        parent_vecs = node_vecs.index_select(dim=1, index=parent_indices.view(-1))  # (batch_size, num_nodes * max_num_parents, hidden_size)

        # Reshape parent vecs
        parent_vecs = parent_vecs.view(parent_vecs_size)  # (batch_size, num_nodes, max_num_parents, hidden_size)

        # Sum parent vecs to get a single parent vector for each node
        parent_vecs = parent_vecs.sum(dim=2)  # (batch_size, num_nodes, hidden_size)

        return parent_vecs  # (batch_size, num_nodes, hidden_size)

    def concat_node_embeddings(self,
                               parent_vecs: torch.FloatTensor,  # (batch_size, num_nodes, hidden_size)
                               node_indices: torch.FloatTensor  # (num_nodes,)
                               ) -> torch.FloatTensor:  # (batch_size, num_nodes, hidden_size + embedding_size)
        """Concatenates node embeddings with the parent vecs."""
        # Get node embeddings
        node_embeddings = self.node_embeddings(node_indices)  # (num_nodes, embedding_size)
        node_embeddings = node_embeddings.unsqueeze(dim=0).repeat(parent_vecs.size(0), 1, 1)  # (batch_size, num_nodes, embedding_size)

        # Concatenate parent vecs and node embeddings
        vecs = torch.cat((parent_vecs, node_embeddings), dim=2)  # (batch_size, num_nodes, hidden_size + embedding_size)

        return vecs  # (batch_size, num_nodes, hidden_size + embedding_size)

    def forward(self, embedding: torch.FloatTensor  # (batch_size, input_size)
                ) -> torch.FloatTensor:  # (batch_size, num_nodes)
        # Get batch size and device
        batch_size = embedding.size(0)

        # Apply input MLP
        embedding = self.input_mlp(embedding)  # (batch_size, hidden_size)

        # Root vector is equal to input embedding
        node_vecs = embedding.unsqueeze(dim=1)  # (batch_size, 1, hidden_size)
        padding = torch.zeros_like(node_vecs)  # (batch_size, 1, hidden_size)
        node_vecs = torch.cat((padding, node_vecs), dim=1)  # (batch_size, 1, hidden_size)

        # Computing
        for depth in range(1, self.dag.max_depth + 1):
            # Get nodes and indices for nodes at this depth, sorted by index
            nodes, node_indices = self.get_nodes_and_indices(depth=depth)  # (num_nodes,)

            # Get parent vecs
            parent_vecs = self.get_parent_vecs(depth=depth, nodes=nodes, node_vecs=node_vecs)  # (batch_size, num_nodes, hidden_size)

            # Get node embeddings and concatenate with parent vecs if needed
            if self.layer_type in ['shared', 'per_depth']:
                vecs = self.concat_node_embeddings(parent_vecs=parent_vecs, node_indices=node_indices)    # (batch_size, num_nodes, hidden_size + embedding_size)
            elif self.layer_type == 'per_node':
                vecs = parent_vecs  # (batch_size, num_nodes, hidden_size)
            else:
                raise ValueError(f'Layer type "{self.layer_type}" is not supported.')

            # Select MLP based on depth if needed
            mlp = self.mlps[depth - 1] if self.layer_type == 'per_depth' else self.mlps[0]

            # Apply MLP
            vecs = vecs.view(-1, vecs.size(2))  # (batch_size * num_nodes, hidden_size [+ embedding_size])
            vecs = mlp(vecs)  # (batch_size * num_nodes, hidden_size)
            vecs = vecs.view((batch_size, len(nodes), self.hidden_size))  # (batch_size, num_nodes, hidden_size)

            # Skip connection
            vecs = parent_vecs + vecs  # (batch_size, num_nodes, hidden_size)

            # Concatenate vecs with node vecs
            node_vecs = torch.cat((node_vecs, vecs), dim=1)  # (batch_size, cumulative_num_nodes, hidden_size)

        # Remove padding
        node_vecs = node_vecs[:, 1:, :]  # (batch_size, total_num_nodes, hidden_size)

        # Final node predictions using node vecs
        node_vecs = self.output_layer(node_vecs)

        return node_vecs
