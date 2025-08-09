import math
import torch
import logging
import pathlib
import argparse

from utils.cfg import get_project_cfg
from utils.embeddings import IREmbeddings
from utils.train import get_corpus_splits

from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GCN

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
    logger.info("Starting data preprocessing")

    split_data = []
    split_len = len(split)
    split_digits = int(math.log10(split_len)) + 1

    for idx, path in enumerate(split):
        logger.info(
            f"[{str(idx).rjust(split_digits)}/{split_len}] Processing path: {path}"
        )

        proj, cfg = get_project_cfg(path)
        embeddings = ir_embed.get_function_embeddings(proj, cfg)
        split_data.extend(embeddings.values())

    logger.info("Data preprocessing complete")

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

    logger.info("Preparing training data")
    training_data = prepare_data_for_split(train, ir_embed)

    logger.info("Preparing test data")
    test_data = prepare_data_for_split(test, ir_embed)

    train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Binary classifier
    model = GCN(
        in_channels=training_data[0].num_features,
        out_channels=2,
        hidden_channels=16,
        num_layers=3,
        add_self_loops=False,
    )
    model.to(ir_embed.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    logger.info("Starting model training")

    for epoch in range(100):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(ir_embed.device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().cpu().item()

    model.eval()

    logger.info("Model training complete")

    correct = 0
    total = 0

    logger.info("Starting model test")

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(ir_embed.device)
            out = model(batch.x, batch.edge_index)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y.view(-1)).sum().item()
            total += batch.y.size(0)

    logger.info("Model testing complete")
    logger.info(f"Test Accuracy: {correct / total:.4f}")


def save_model(model):
    # TODO: save model one we have one for good accuracy to use in the actual tool
    pass


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
