import pathlib
from collections.abc import Iterator

import angr
import networkx as nx
from angr.knowledge_plugins.cfg import CFGNode
from angr.knowledge_plugins.functions import Function
from angr.analyses.cfg import CFGFast
from pyvex.stmt import AbiHint, IMark, IRStmt, NoOp

from .graph import insert_node_attributes


def lift_stmt_ir(
    cfg: nx.DiGraph,
) -> Iterator[tuple[CFGNode, list[IRStmt] | None]]:
    """
    Iterator that yields the IR of each `IRStmt` of each `CFGNode` in `cfg`
    """

    def is_marker(stmt: IRStmt) -> bool:
        """
        Checks if `stmt` is a non-semantical statement
        """
        return isinstance(stmt, (AbiHint, IMark, NoOp))

    for node in cfg.nodes():
        block = cfg.nodes[node].get("block")

        if block:
            if block.vex.has_statements:
                stmts = [s for s in block.vex.statements if not is_marker(s)]
                yield (node, stmts)
            else:
                yield (node, None)
        else:
            yield (node, None)


def get_project_cfg(file: pathlib.Path) -> tuple[angr.Project, CFGFast]:
    """
    Returns the angr project and CFG of `file`
    """
    project = angr.Project(file, auto_load_libs=False)
    cfg = project.analyses.CFGFast(
        # Force complete scan to get as much information as possible
        force_complete_scan=True,
        force_smart_scan=False,
        resolve_indirect_jumps=True,
        normalize=True,
    )

    return project, cfg


def get_sub_cfgs(cfg: CFGFast) -> Iterator[tuple[Function, nx.DiGraph]]:
    """
    Iterator that yields a `nx.DiGraph` corresponding to a subgraph of `cfg`
    """

    for func in cfg.kb.functions.values():
        subgraph = func.transition_graph
        if len(subgraph.edges()) > 0:
            for node in subgraph:
                cfgnode = cfg.model.get_any_node(node.addr)
                insert_node_attributes(subgraph, {node: {"name": cfgnode.name}})
                insert_node_attributes(
                    subgraph, {node: {"block": cfgnode.block}}
                )

            yield func, subgraph
