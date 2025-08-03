import sys
import angr
import logging
import pathlib
import networkx

from collections.abc import Generator


# Silence, fools
# We know the symbol was allocated without a known size
# And we can't do anything about it!
logger = logging.getLogger("cle.backends.externs")
logger.setLevel(logging.ERROR)

logger = logging.getLogger("cle.loader")
logger.setLevel(logging.ERROR)


def crawl(path: pathlib.Path, pattern: str) -> Generator[pathlib.Path]:
    """
    Generator that yields `pathlib.Path`s in `path` that match `pattern`
    """
    for file in path.rglob(pattern):
        yield file


def lift(cwe_id: str, file: pathlib.Path) -> Generator[networkx.DiGraph]:
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

    To address this, we preemptively perform the same check, and standardize
    any identified difference
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

        # The CFG contains a weakly connected component for each test case
        wcc = networkx.weakly_connected_components(src_cfg.model.graph)
        sub_cfgs = [src_cfg.model.graph.subgraph(i).copy() for i in wcc]

        for sub_cfg in sub_cfgs:
            try:
                edge_attrs = set(
                    list(next(iter(sub_cfg.edges(data=True)))[-1].keys())
                )
                for idx, (u, v, attr_dict) in enumerate(
                    sub_cfg.edges(data=True)
                ):
                    key_set = set(attr_dict.keys())
                    if key_set != edge_attrs:
                        new_attr_dict = attr_dict

                        if len(key_set) < len(edge_attrs):
                            # Often, edges with `jumpkind = Ijk_Boring` lack `stmt_idx`
                            # Those with `stmt_idx` seem to always have it set to -2
                            for diff in edge_attrs.difference(key_set):
                                new_attr_dict[diff] = -2

                        elif len(key_set) > len(edge_attrs):
                            # Removes the offending difference
                            # TODO: Fix last edge against which all others are compared
                            for diff in key_set.difference(edge_attrs):
                                del new_attr_dict[diff]

                        # Update edge with new attributes
                        networkx.set_edge_attributes(
                            sub_cfg, {(u, v): new_attr_dict}
                        )

            except StopIteration:
                # NOTE: CWE-703!
                # TODO: Handle this in a less hacky fashion
                pass  # nosec B110

            yield sub_cfg


def main():
    if len(sys.argv) != 3:
        print("Usage: python rip_cfgs.py [CWE-ID] [PATH]")
        sys.exit()

    cwe_id = sys.argv[1]
    path = pathlib.Path(sys.argv[2])

    for m in crawl(path, f"{cwe_id}*/**/main_linux.o"):
        for cfg in lift(cwe_id, m):
            # TODO: What now?
            continue


if __name__ == "__main__":
    main()
