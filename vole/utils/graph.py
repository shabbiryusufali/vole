import pathlib
from collections.abc import Iterator

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


def get_digraph_source_node(graph: nx.DiGraph) -> any:
    """
    Returns the source of `graph` if it exists
    Each CFG is expected to have at most one source
    """
    sources = [
        n for n in graph.nodes() if graph.in_degree(n) == 0 and graph.out_degree(n) > 0
    ]
    return sources[0] if sources else None


def traverse_digraph(graph: nx.DiGraph) -> Iterator[any]:
    """
    Iterator that yields nodes in `graph`
    """
    for node, neighbours in graph.adjacency():
        for neighbour in neighbours.keys():
            yield neighbour


def extract_subgraphs(graph: nx.Graph) -> list[nx.Graph]:
    """
    Extracts subgraphs of `graph` belonging to separate weakly connected components
    NOTE: This is our best heuristic to identify CFGS of independent functions
    """
    wcc = [graph.subgraph(i).copy() for i in nx.weakly_connected_components(graph)]

    if len(wcc) > 1:
        return wcc

    # NOTE: It seems that many of the test cases do not satisfy the weakly connected property
    # TODO: Another heuristic for this???
    # attempting to get subgraphs using function names (this doesn't care for called functions which might be an issue idk lol)
    function_groups = {}

    for node in graph.nodes():
        func_name = get_function_name_from_node(node)
        if func_name not in function_groups:
            function_groups[func_name] = []
        function_groups[func_name].append(node)

    if len(function_groups) > 1:
        subgraphs = []
        for func_name, nodes in function_groups.items():
            if nodes:
                subgraph = graph.subgraph(nodes).copy()
                subgraphs.append(subgraph)
        return subgraphs
    # TODO: Remove this return once we decide
    return wcc

def get_function_name_from_node(node: any) -> str | None:

    if hasattr(node, 'function') and node.function:
        return node.function.name
    elif hasattr(node, 'name') and node.name:
        return node.name
    return None

def normalize_edge_attributes(graph: nx.Graph) -> None:
    """
    Ensures that all edges have consistent attributes
    """
    try:
        # The keys against which attributes of all other edges will be compared
        edge_attrs = set(list(next(iter(graph.edges(data=True)))[-1].keys()))

        for idx, (u, v, attrs) in enumerate(graph.edges(data=True)):
            key_set = set(attrs.keys())

            if key_set == edge_attrs:
                continue

            new_attrs = attrs

            if len(key_set) < len(edge_attrs):
                # Inserts a default value for the missing attribute
                for diff in edge_attrs.difference(key_set):
                    new_attrs[diff] = None

            elif len(key_set) > len(edge_attrs):
                # Removes the offending difference
                # TODO: Shouldn't we instead correct `edge_attrs`?
                for diff in key_set.difference(edge_attrs):
                    del new_attrs[diff]

            # Update edge with updated attributes
            nx.set_edge_attributes(graph, {(u, v): new_attrs})

    except StopIteration:
        # NOTE: CWE-703!
        # TODO: Handle this in a less hacky fashion
        pass  # nosec B110


def insert_node_attributes(graph: nx.Graph, attrs: dict[any, dict[str, any]]) -> None:
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
