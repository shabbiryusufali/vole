import pathlib

import matplotlib
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import convert


EDGE_TYPES = {"transition": 0, "call": 1, "fake_return": 2}


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
    for idx, (u, v, attrs) in enumerate(graph.edges(data=True)):
        new_attrs = {}

        # Only care about `type` and `outside` for now
        new_attrs["type"] = EDGE_TYPES.get(attrs.get("type", "transition"), 0)
        new_attrs["outside"] = 1 if attrs.get("outside", False) else 0

        attrs.clear()
        nx.set_edge_attributes(graph, {(u, v): new_attrs})


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
    NOTE: Node labels must be relabelled
    """
    normalize_edge_attributes(graph)
    nx.convert_node_labels_to_integers(graph)
    return convert.from_networkx(graph)
