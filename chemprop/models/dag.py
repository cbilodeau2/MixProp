"""Contains a model which can make predictions on the nodes of a DAG."""
from typing import Any, Set

import networkx
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

        self.add_edges_from(set.union(*[{(p, node) for p in node.parents} |
                                        {(node, c) for c in node.children}
                                        for node in nodes]))
        if not networkx.is_directed_acyclic_graph(self):
            raise ValueError('Node relationships do not form a DAG.')

        self.root = self.get_root()

    def get_root(self) -> Node:
        """Gets the root Node from a set of Nodes, raising a ValueError if no single Node is the root."""
        nodes_without_parents = {node for node in self.nodes if len(self.in_edges(node)) == 0}

        if len(nodes_without_parents) == 0:
            raise ValueError('No root node found, all nodes have parents.')

        if len(nodes_without_parents) > 1:
            raise ValueError('No root node found, multiple nodes have no parents.')

        return nodes_without_parents.pop()
