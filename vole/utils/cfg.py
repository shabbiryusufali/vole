import pathlib
from collections.abc import Iterator

import angr
import networkx as nx
from angr.knowledge_plugins.cfg import CFGNode
from angr.knowledge_plugins.functions import Function
from angr.analyses.cfg import CFGFast
from pyvex.stmt import AbiHint, IMark, IRStmt, NoOp

from .graph import insert_node_attributes


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
            yield func, subgraph
