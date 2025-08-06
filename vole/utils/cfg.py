import pathlib
import sys
from collections.abc import Iterator

import angr
import networkx as nx
import torch
import utils.model_OTA  # NOTE: MUST be imported this way
from angr.knowledge_plugins.cfg import CFGNode
from angr.analyses.cfg import CFGFast
from pyvex.stmt import AbiHint, IMark, IRStmt, NoOp

from .graph import insert_node_attributes


def lift_stmt_ir(
    cfg: nx.DiGraph
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


def get_program_cfg(file: pathlib.Path) -> CFGFast:
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


def get_sub_cfgs(cfg: CFGFast) -> Iterator[nx.DiGraph]:
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

            yield subgraph


def vectorize_stmt_ir(stmts: list[IRStmt] | None) -> list[list[int]]:
    """
    Converts the IR for each statement in `stmts` to a vector representation using VexIR2Vec
    """
    if not stmts:
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    here = pathlib.Path(__file__).parent.resolve()
    model_path = pathlib.Path(here / "../models/vexir2vec.model")

    # Patch the resolution of the model's source at runtime
    sys.modules["model_OTA"] = utils.model_OTA

    # TODO: Figure out if it's possible to load the model with weights_only=True
    vexir2vec = torch.load(model_path, map_location=device, weights_only=False)
    vexir2vec.eval()

    # TODO: Format each statement in `stmts` to be run in `vexir2vec`
    # NOTE: For usage, see: https://github.com/IITH-Compilers/VexIR2Vec/blob/43538167644db81cbfda89716c113b483aa9fd06/experiments/diffing/v2v_diffing.py#L82-L344

    # TODO: Return vector embeddings for each statement
    return [1]
