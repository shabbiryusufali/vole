import angr
import pathlib
import networkx as nx

from .graph import (
    traverse_digraph,
    extract_subgraphs,
    normalize_edge_attributes,
    insert_node_attributes,
)

from angr.knowledge_plugins.cfg import CFGNode
from collections.abc import Iterator


def lift_block_ir(cfg: nx.DiGraph) -> tuple[CFGNode, str]:
    """
    Iterator that yields the IR of each `CFGNode` in `cfg`
    """
    for node in traverse_digraph(cfg):
        if node.block:
            yield (node, str(node.block.vex))


def lift_stmt_ir(cfg: nx.DiGraph) -> tuple[CFGNode, str]:
    """
    Iterator that yields the IR of each `IRStmt` of each `CFGNode` in `cfg`
    """
    for node in traverse_digraph(cfg):
        if node.block:
            for stmt in node.block.vex.statements:
                yield (node, str(stmt))


def get_program_cfg(file: pathlib.Path):
    """
    Returns the CFG of `file`
    """
    project = angr.Project(file, auto_load_libs=False)
    cfg = project.analyses.CFGFast(
        # Force complete scan to get as much information as possible
        force_complete_scan=True,
        force_smart_scan=False,
        resolve_indirect_jumps=True,
        normalize=True,
    )

    return cfg


def get_sub_cfgs(cfg: nx.DiGraph) -> Iterator[nx.DiGraph]:
    """
    Iterator that yields a `nx.DiGraph` corresponding to a subgraph of `cfg`
    """
    for sub_cfg in extract_subgraphs(cfg.model.graph):
        normalize_edge_attributes(sub_cfg)

        yield sub_cfg


def vectorize_node_ir(cfg: nx.DiGraph) -> None:
    """
    Converts the IR for each node in the `cfg` to a vector representation and inserts it into the node as an attribute
    """
    node_attrs = {k: v for k, v in lift_block_ir(cfg)}

    # TODO: Convert node_attrs.values() to vectors
    # NOTE: See utils/train.py

    insert_node_attributes(cfg, "ir_vec", node_attrs)
