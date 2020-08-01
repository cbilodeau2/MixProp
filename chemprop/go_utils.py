"""Utility functions for working with the Gene Ontology (GO) hierarchy."""
from typing import List

from goatools.base import download_go_basic_obo
from goatools.obo_parser import GODag

from chemprop.dag import Node, RootedDAG


def go_dag_to_rooted_dag(go_dag: GODag, go_ids: List[str]) -> RootedDAG:
    """Given a GODag and a list of GO ID, builds a RootedDAG object containing the provided GO IDs."""
    go_id_to_node = {go_id: Node(go_id) for go_id in set(go_ids)}

    for go_id, go_node in go_id_to_node.items():
        go_node.parents = {go_id_to_node[parent_go_id.item_id] for parent_go_id in go_dag[go_id].parents
                           if parent_go_id.item_id in go_ids}
        go_node.children = {go_id_to_node[child_go_id.item_id] for child_go_id in go_dag[go_id].children
                            if child_go_id.item_id in go_ids}

    rooted_dag = RootedDAG(set(go_id_to_node.values()))

    return rooted_dag


def load_go_dag(go_obo_path: str, go_ids: List[str]) -> RootedDAG:
    """Given a list of GO IDs, builds a RootedDag object containing the provided GO IDs"""
    go_obo_fname = download_go_basic_obo(go_obo_path)
    go_dag = GODag(go_obo_fname)
    rooted_dag = go_dag_to_rooted_dag(go_dag=go_dag, go_ids=go_ids)

    assert rooted_dag.number_of_nodes() == len(go_ids)

    return rooted_dag
