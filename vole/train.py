import logging
import pathlib
import argparse
import torch
from torch_geometric.nn import GCN
from torch_geometric.loader import DataLoader

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


def prepare_training_data(split: list[pathlib.Path]) -> list:
    training_data = []

    for idx, path in enumerate(split):
        cfg = get_program_cfg(path)

        for sub_cfg in get_sub_cfgs(cfg):
            source = get_digraph_source_nodes(sub_cfg)[0]
            name = sub_cfg.nodes[source].get("name")

            # TODO: More granular labelling
            if not name:
                label = -1  # Invalid
            elif "bad" in name:
                label = 1  # Bad (i.e. vulnerable)
            elif "good" in name:
                label = 0  # Good

            if label == -1:
                continue

            for node, stmts_ir in lift_stmt_ir(sub_cfg):
                # Insert features as node attributes
                # This ensures the values are preserved by torch later
                stmts_vec = vectorize_stmt_ir(stmts_ir)
                insert_node_attributes(sub_cfg, {node: {"label": label}})
                insert_node_attributes(sub_cfg, {node: {"ir_vec": stmts_vec}})

            training_data.append(to_torch_data(sub_cfg))

    return training_data


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

    # TODO: Train GCN on `training_data`
    # possible model training code (i can't test it rn | model assuming graph level labelling)
    training_data = prepare_training_data(train)
    test_data = prepare_training_data(test)

    train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # in channels needs to be changed most likely
    # out channels is 2 for binary classification
    model = GCN(in_channels=training_data[0].num_features, out_channels=2, num_layers=3)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(100):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            # might need to do pooling for graph level classification (GCN might handle it)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y.view(-1)).sum().item()
            total += batch.y.size(0)
    print(f"Test Accuracy: {correct / total:.4f}")

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
