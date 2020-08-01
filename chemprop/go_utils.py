"""Utility functions for working with the Gene Ontology (GO) hierarchy."""
from typing import Set

from goatools.obo_parser import GODag

from chemprop.models import Node, RootedDAG


def go_to_dag(go_dag: GODag, go_ids: Set[str]) -> RootedDAG:
    """Given a GODag and a list of go_ids, builds a DAG object containing the provided go_ids."""
    go_id_to_node = {go_id: Node(go_id) for go_id in go_ids}

    for go_id, go_node in go_id_to_node.items():
        go_node.parents = {go_id_to_node[parent_go_id.item_id] for parent_go_id in go_dag[go_id].parents
                           if parent_go_id.item_id in go_ids}
        go_node.children = {go_id_to_node[child_go_id.item_id] for child_go_id in go_dag[go_id].children
                            if child_go_id.item_id in go_ids}

    dag = RootedDAG(set(go_id_to_node.values()))

    return dag
