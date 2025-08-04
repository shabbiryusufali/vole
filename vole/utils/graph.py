import pathlib
import networkx as nx
import matplotlib

from collections.abc import Iterator


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
    """
    return [graph.subgraph(i).copy() for i in nx.weakly_connected_components(graph)]


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


def insert_node_attributes(
    graph: nx.Graph, attr_key: str, attrs: dict[any, any]
) -> None:
    """
    Inserts `attrs` for each corresponding node in the `graph`
    """
    for node in graph.nodes():
        attr_val = attrs.get(node, None)
        nx.set_node_attributes(graph, {node: {attr_key: attr_val}})
