import pathlib

import matplotlib
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import convert


def save_graph(graph: nx.Graph, path: pathlib.Path) -> None:
    """
    Saves `graph` as a figure to `path`
    """
    nx.draw(graph, with_labels=True, font_size=5)
    matplotlib.pyplot.savefig(path)
    matplotlib.pyplot.clf()


def normalize_edge_attributes(graph: nx.Graph) -> None:
    """
    Ensures that all edges have consistent attributes
    """
    try:
        # The keys against which attributes of all other edges will be compared
        edge_attrs = set(list(next(iter(graph.edges(data=True)))[-1].keys()))

        for idx, (u, v, attrs) in enumerate(graph.edges(data=True)):
            new_attrs = {key: attrs.get(key) for key in edge_attrs}
            attrs.clear()
            nx.set_edge_attributes(graph, {(u, v): new_attrs})

    except StopIteration:
        # NOTE: CWE-703!
        # TODO: Handle this in a less hacky fashion
        pass  # nosec B110


def insert_node_attributes(graph: nx.Graph, attrs: dict) -> None:
    """
    Inserts `attrs` into `graph
    NOTE: `attrs` should be in the form `{node: {key: value}}` where
    - `node` is the node to insert the attribute into
    - `key` is the name of the attribute
    - `value` is the value of the attribute
    """
    nx.set_node_attributes(graph, attrs)


def to_torch_data(graph: nx.Graph) -> Data:
    """
    Convert `graph` to an instance of `torch_geometric.data.Data`
    NOTE: Edges must be normalized in order for `from_networkx` to play nice
    """
    normalize_edge_attributes(graph)
    return convert.from_networkx(graph)
