import logging
import pathlib
import argparse

from utils.cfg import get_project_cfg
from utils.embeddings import IREmbeddings
from utils.train import get_corpus_splits

# Silence angr
logger = logging.getLogger("cle.backends.externs")
logger.setLevel(logging.ERROR)

logger = logging.getLogger("cle.loader")
logger.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def prepare_data_for_split(
    split: list[pathlib.Path], ir_embed: IREmbeddings
) -> list:
    split_data = []

    for path in split:
        proj, cfg = get_project_cfg(path)
        embeddings = ir_embed.get_function_embeddings(proj, cfg)
        split_data.extend(embeddings.values())

    return split_data


def train_gcn(cwe_id: str, path: pathlib.Path):
    """
    Trains a `torch_geometric.nn.models.GCN` from SARD test case binaries
    matching `cwe_id` in `path`
    """
    train, test = get_corpus_splits(cwe_id, path)

    if not all((train, test)):
        logger.info(
            f"""
            CWE-ID `{cwe_id}` and path `{path}` yielded no results.
            Check that `path` contains the compiled test cases.
            """
        )

    ir_embed = IREmbeddings()

    prepare_data_for_split(train, ir_embed)

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
