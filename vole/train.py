import math
import torch
import logging
import pathlib
import argparse

from utils.cfg import get_project_cfg
from utils.embeddings import IREmbeddings
from utils.io import get_corpus_splits

from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GCN

# Silence angr
logger = logging.getLogger("cle.backends.externs")
logger.setLevel(logging.ERROR)

logger = logging.getLogger("cle.loader")
logger.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


PARENT = pathlib.Path(__file__).parent.resolve()


def prepare_data_for_split(
    split: list[pathlib.Path], ir_embed: IREmbeddings
) -> list:
    logger.info("Starting data preprocessing")

    split_data = []
    split_len = len(split)
    split_digits = int(math.log10(split_len)) + 1

    for idx, path in enumerate(split):
        logger.info(
            f"[{str(idx + 1).rjust(split_digits)}/{split_len}] Processing path: {path}"
        )

        proj, cfg = get_project_cfg(path)
        embeddings = ir_embed.get_function_embeddings(proj, cfg)
        split_data.extend(embeddings.values())

    logger.info("Data preprocessing complete")

    return split_data


def do_training(split: list[pathlib.Path], ir_embed: IREmbeddings, device):
    logger.info("Preparing training data")

    train_data = prepare_data_for_split(split, ir_embed)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # Binary classifier
    model = GCN(
        in_channels=train_data[0].num_features,
        out_channels=2,
        hidden_channels=64,
        num_layers=3,
        dropout=0.5,
        add_self_loops=False,
    )
    model.to(device)

    labels = torch.cat([data.y for data in train_data])
    labels = labels.to(device)

    class_counts = torch.bincount(labels)
    class_counts = class_counts.to(device)

    weights = class_counts.sum() / (len(class_counts) * class_counts.float())
    weights = weights.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    criterion = criterion.to(device)

    logger.info("Starting model training")

    model.train()
    for epoch in range(100):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().cpu().item()

    torch.cuda.empty_cache()

    logger.info("Model training complete")

    return model


def do_testing(
    split: list[pathlib.Path], ir_embed: IREmbeddings, device, model: GCN
):
    logger.info("Preparing test data")
    test_data = prepare_data_for_split(split, ir_embed)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    correct = 0
    total = 0

    logger.info("Starting model test")

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y.view(-1)).sum().item()
            total += batch.y.size(0)

    torch.cuda.empty_cache()

    logger.info("Model testing complete")
    logger.info(f"Test Accuracy: {correct / total:.4f}")

    return model


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
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ir_embed = IREmbeddings(device, train=True)

    model = do_training(train, ir_embed, device)
    model = do_testing(test, ir_embed, device, model)

    out_path = pathlib.Path(PARENT / f"./models/{cwe_id.upper()}.model")
    logger.info(f"Saving model to {out_path}")
    save_model(model, out_path)


def save_model(model, out_path: pathlib.Path) -> None:
    torch.save(model.state_dict(), out_path)


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
