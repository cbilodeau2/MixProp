"""Converts a GO hierarchy into a DAG."""
from typing import List

from goatools.obo_parser import GODag

from .dag import DAG, Node


def go_to_dag(go_dag: GODag, go_ids: List[str]) -> DAG:
    """Given a GODag and a list of go_ids, builds a DAG object containing the provided go_ids."""
    go_id_to_node = {go_id: Node(go_id) for go_id in set(go_ids)}

    for go_id, go_node in go_id_to_node.items():
        node = go_id_to_node[go_id]
        go_node.parents =
