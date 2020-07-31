"""Contains a model which can make predictions on the nodes of a DAG."""
from typing import Any, List, Set

import networkx as nx
from networkx import DiGraph


class Node:
    def __init__(self,
                 node_id: str,
                 parents: Set['Node'] = None,
                 children: Set['Node'] = None):
        self.node_id = node_id
        self.parents = parents if parents is not None else set()
        self.children = children if children is not None else set()

    def __str__(self) -> str:
        return f'Node({self.node_id})'

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        """Node hash is based on self ID, parent IDs, and children IDs."""
        return hash(self.node_id)

    def __eq__(self, other: Any) -> bool:
        """Nodes are equal if they, their parents, and they children all have matching IDs."""
        if not isinstance(other, Node):
            return False

        return self.node_id == other.node_id


class RootedDAG(DiGraph):
    def __init__(self, nodes: Set[Node]):
        super(RootedDAG, self).__init__()

        self.add_edges_from(set.union(*[{(p.node_id, node.node_id) for p in node.parents} |
                                        {(node.node_id, c.node_id) for c in node.children}
                                        for node in nodes]))

        if not nx.is_directed_acyclic_graph(self):
            raise ValueError('Node relationships do not form a DAG.')

        self.root = self.get_root()
        self._max_depth = None

        self._node_to_index = {}
        for node in self.nodes:
            self._node_to_index[node] = len(self._node_to_index)

    def get_root(self) -> str:
        """Gets the root Node from a set of Nodes, raising a ValueError if no single Node is the root."""
        nodes_without_parents = {node for node in self.nodes if len(self.in_edges(node)) == 0}

        if len(nodes_without_parents) == 0:
            raise ValueError('No root node found, all nodes have parents.')

        if len(nodes_without_parents) > 1:
            raise ValueError('No root node found, multiple nodes have no parents.')

        return nodes_without_parents.pop()

    def depth(self, node: str) -> int:
        """Gets the depth of a node (i.e., longest distance from the root)."""
        if 'depth' not in self.nodes[node]:
            path_lengths = [len(path) for path in nx.algorithms.all_simple_paths(self, self.root, node)]
            max_path_length = max(path_lengths) if len(path_lengths) > 0 else 1
            self.nodes[node]['depth'] = max_path_length - 1

        return self.nodes[node]['depth']

    @property
    def max_depth(self) -> int:
        """Gets the maximum depth of any node."""
        if self._max_depth is None:
            self._max_depth = max(self.depth(node) for node in self.nodes)

        return self._max_depth

    def node_to_index(self, node: str) -> int:
        """Returns the index for a node."""
        return self._node_to_index[node]
