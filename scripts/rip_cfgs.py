import sys
import angr
import logging
import pathlib
import networkx
import matplotlib

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


def save_graph(graph: networkx.Graph, path: pathlib.Path) -> None:
    """
    Saves `graph` as a figure to `path`
    NOTE: Useful for sanity checking purposes :)
    """
    networkx.draw(graph, with_labels=True, font_size=5)
    matplotlib.pyplot.savefig(path)
    matplotlib.pyplot.clf()


def lift(cwe_id: str, file: pathlib.Path) -> None:
    """
    Resolves subgraphs of CFG corresponding to test cases
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
        # Extract the subgraphs
        wcc = networkx.weakly_connected_components(src_cfg.model.graph)
        sub_cfgs = [src_cfg.model.graph.subgraph(i).copy() for i in wcc]

        for cfg in sub_cfgs:
            source = [n for n in cfg.nodes() if cfg.in_degree(n) == 0][0]

            # [ ] TODO (optional): Convert subgraphs to instances of `CFGFast` (better for serialization)
            # [ ] TODO: Serialize and save to disk


def main():
    if len(sys.argv) != 3:
        print("Usage: python rip_cfgs.py [CWE-ID] [PATH]")
        sys.exit()

    cwe_id = sys.argv[1]
    path = pathlib.Path(sys.argv[2])

    for m in crawl(path, f"{cwe_id}*/**/main_linux.o"):
        lift(cwe_id, m)


if __name__ == "__main__":
    main()
