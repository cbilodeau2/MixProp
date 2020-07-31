"""Contains a model which can make predictions on the nodes of a DAG."""
from typing import Any, List, Tuple


class Node:
    def __init__(self,
                 node_id: str,
                 is_root: bool = False,
                 parents: List['Node'] = None,
                 children: List['Node'] = None):
        self.node_id = node_id
        self.is_root = is_root
        self.parents = parents if parents is not None else []
        self.children = children if children is not None else []

    def __str__(self) -> str:
        return f'Node({self.node_id})'

    def _get_self_parent_and_children_ids(self) -> Tuple[str, Tuple[str], Tuple[str]]:
        """Gets a tuple of the self ID, a sorted tuple of parent IDs, and a sorted tuple of child IDs."""
        return (self.node_id,
                tuple(sorted(p.node_id for p in self.parents)),
                tuple(sorted(c.node_id for c in self.children)))

    def __hash__(self) -> int:
        """Node hash is based on self ID, parent IDs, and children IDs."""
        return hash(self._get_self_parent_and_children_ids())

    def __eq__(self, other: Any):
        """Nodes are equal if they, their parents, and they children all have matching IDs."""
        if not isinstance(other, Node):
            return False

        return hash(self) == hash(other)


class DAG(dict):
    def __init__(self, nodes: List[Node]):
        super(DAG, self).__init__()
        if len({node.node_id for node in nodes}) != len(nodes):
            raise ValueError('All nodes must have unique IDs.')

        for node in nodes:
            self[node.node_id] = node

    @staticmethod
    def _check_if_dag(root: Node):
        assert root.is_root

        visited = set()
        stack = [root]

        while len(stack):
            pass
