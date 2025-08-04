import angr
import logging
import pathlib
import argparse
import networkx as nx

from utils.io import crawl
from utils.graph import (
    extract_subgraphs,
    get_digraph_source_node,
    normalize_edge_attributes,
    traverse_digraph,
)

from collections.abc import Iterator


# Silence, fools
# We know the symbol was allocated without a known size
# And we can't do anything about it!
logger = logging.getLogger("cle.backends.externs")
logger.setLevel(logging.ERROR)

logger = logging.getLogger("cle.loader")
logger.setLevel(logging.ERROR)


def output_ir(cfg: nx.DiGraph, path: pathlib.Path) -> None:
    source: angr.knowledge_plugins.cfg.CFGNode = get_digraph_source_node(cfg)

    if not source:
        return

    ir = "\n".join(
        [str(node.block.vex) for node in traverse_digraph(cfg) if node.block]
    )

    with open(f"{path}/{source.name}.txt", "w") as f:
        f.write(ir)


def lift(
    cwe_id: str, file: pathlib.Path, should_output_ir: bool
) -> Iterator[nx.DiGraph]:
    """
    Resolves subgraphs of CFG corresponding to test cases

    The edges of each subgraph must be normalized to play nicely with
    `torch_geometric.utils.convert.from_networkx`.

    Why?
        1. `pyvex` inconsistently assigns edge attributes; sometimes edges are
            missing `stmt_idx`
        2. `torch_geometric.utils.convert.from_networkx` determines which edge
            attributes are "standard" by inspecting the last edge in the graph,
            then checks that all other edges have the same attributes
    """
    # Load each source binary
    for o in crawl(file.parent, f"**/{cwe_id}*.o"):
        # Generate CFG for entire binary
        src_proj = angr.Project(o, auto_load_libs=False)
        src_cfg = src_proj.analyses.CFGFast(
            # Force complete scan to get as much information as possible
            force_complete_scan=True,
            force_smart_scan=False,
            resolve_indirect_jumps=True,
            normalize=True,
        )

        for sub_cfg in extract_subgraphs(src_cfg.model.graph):
            normalize_edge_attributes(sub_cfg)

            if should_output_ir:
                output_ir(sub_cfg, file.parent)

            yield sub_cfg


def parse() -> dict:
    parser = argparse.ArgumentParser(
        prog="rip.py",
        description="""
            Rips CFGs and/or IR from compiled SARD test cases
        """,
    )

    parser.add_argument("CWE-ID", type=str)
    parser.add_argument("path", type=pathlib.Path)
    parser.add_argument(
        "-i", "--ir", action="store_true", help="Whether or not to output IR"
    )

    return vars(parser.parse_args())


def main():
    args = parse()

    cwe_id = args.get("CWE-ID")
    path = args.get("path")
    should_output_ir = args.get("ir")

    for m in crawl(path, f"{cwe_id}*/**/main_linux.o"):
        for cfg in lift(cwe_id, m, should_output_ir):
            # TODO: What now?
            continue


if __name__ == "__main__":
    main()
