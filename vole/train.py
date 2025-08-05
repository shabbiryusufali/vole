import logging
import pathlib
import argparse

from utils.cfg import get_program_cfg, get_sub_cfgs, lift_stmt_ir, vectorize_stmt_ir
from utils.graph import get_digraph_source_nodes, insert_node_attributes, to_torch_data
from utils.train import get_corpus_splits

# Silence angr
logger = logging.getLogger("cle.backends.externs")
logger.setLevel(logging.ERROR)

logger = logging.getLogger("cle.loader")
logger.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def prepare_data_for_split(split: list[pathlib.Path]) -> list:
    split_data = []

    for idx, path in enumerate(split):
        cfg = get_program_cfg(path)

        for sub_cfg in get_sub_cfgs(cfg):
            source = get_digraph_source_nodes(sub_cfg)[0]
            name = sub_cfg.nodes[source].get("name")

            # TODO: More granular labelling
            if not name:
                label = -1  # Invalid
            elif "bad" in name:
                label = 0  # Bad (i.e. vulnerable)
            elif "good" in name:
                label = 1  # Good

            for node, stmts_ir in lift_stmt_ir(sub_cfg):
                # Insert features as node attributes
                # This ensures the values are preserved by torch later
                stmts_vec = vectorize_stmt_ir(stmts_ir)
                insert_node_attributes(sub_cfg, {node: {"label": label}})
                insert_node_attributes(sub_cfg, {node: {"ir_vec": stmts_vec}})

            split_data.append(to_torch_data(sub_cfg))

    return split_data


def train_gcn(cwe_id: str, path: pathlib.Path):
    """
    Trains a `torch_geometric.nn.models.GCN` from SARD test case binaries
    matching `cwe_id` in `path`
    """
    train, test, evaluation = get_corpus_splits(cwe_id, path)

    if not all((train, test, evaluation)):
        logger.info(
            f"""
            CWE-ID `{cwe_id}` and path `{path}` yielded no results.
            Check that `path` contains the compiled test cases.
            """
        )

    # training_data = prepare_data_for_split(train)

    # TODO: Train GCN on `training_data`


def parse() -> dict:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="""
            Train a PyTorch neural network on SARD test cases
        """,
    )

    parser.add_argument("CWE-ID", type=str)
    parser.add_argument("path", type=pathlib.Path)

    return vars(parser.parse_args())


def main():
    args = parse()
    train_gcn(args.get("CWE-ID"), args.get("path"))


if __name__ == "__main__":
    main()
