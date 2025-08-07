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
        split_data.extend(embeddings)

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

    # TODO: Train GCN on `training_data`
    # possible model training code (i can't test it rn | model assuming graph level labelling)
    training_data = prepare_data_for_split(train)
    test_data = prepare_data_for_split(test)

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
