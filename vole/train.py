import sys
import math
import json
import torch
import optuna
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


def do_training(model: GCN, optimizer) -> None:
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()


def do_testing(model: GCN) -> list[float]:
    correct = 0
    precision_num = 0
    precision_denom = 0
    recall_num = 0
    recall_denom = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            pred = out.argmax(dim=1)
            print(f"Predictions: {pred}, Labels: {batch.y.view(-1)}")
            true_positive = ((pred == batch.y.view(-1) & pred == 1)).sum().item()
            true_negative = ((pred == batch.y.view(-1) & pred == 0)).sum().item()
            false_positive = ((pred == 1) & (
                batch.y.view(-1) == 0
            )).sum().item()
            false_negative = ((pred == 0) & (
                batch.y.view(-1) == 1
            )).sum().item()
            precision_num += true_positive
            precision_denom += true_positive + false_positive
            recall_num += true_positive
            recall_denom += true_positive + false_negative
            correct += true_positive + true_negative
            total += batch.y.size(0)

    accuracy = (correct / total) if total > 0 else 0
    precision = (precision_num / precision_denom) if precision_denom > 0 else 0
    recall = (recall_num / recall_denom) if recall_denom > 0 else 0
    f1_score = 0
    if precision + recall != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return [accuracy, precision, recall, f1_score]


def objective(trial):
    hidden_channels = trial.suggest_int("hidden_channels", 16, 128, step=16)
    num_layers = trial.suggest_int("num_layers", 1, 6)
    dropout = trial.suggest_float("dropout", 1e-1, 1e0, log=True)

    model = GCN(
        in_channels=train_data[0].num_features,
        out_channels=2,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        add_self_loops=False,
    ).to(device)

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical(
        "optimizer", ["Adam", "RMSprop", "SGD"]
    )
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    for epoch in range(100):
        do_training(model, optimizer)
        [accuracy, precision, recall, f1_score] = do_testing(model)

        trial.report(accuracy, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        trial.set_user_attr("epoch", epoch)
        trial.set_user_attr("precision", precision)
        trial.set_user_attr("recall", recall)
        trial.set_user_attr("f1_score", f1_score)

    trial.set_user_attr("state", model.state_dict())

    return accuracy


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


if __name__ == "__main__":
    args = parse()
    cwe_id = args.get("CWE-ID")
    path = args.get("path")

    train, test = get_corpus_splits(cwe_id, path)
    if not all((train, test)):
        logger.info(
            f"""
            CWE-ID `{cwe_id}` and path `{path}` yielded no results.
            Check that `path` contains the compiled test cases.
            """
        )
        sys.exit(1)

    # NOTE: `device` accessed above
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ir_embed = IREmbeddings(device, train=True)

    # NOTE: `train_data` and `train_loader` accessed above
    logger.info("Preparing training data")
    train_data = prepare_data_for_split(train, ir_embed)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # NOTE: `criterion` accessed above
    labels = torch.cat([data.y for data in train_data]).to(device)
    class_counts = torch.bincount(labels).to(device)
    weights = class_counts.sum() / (
        len(class_counts) * class_counts.float()
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights).to(device)

    # NOTE: `test_loader` accessed above
    logger.info("Preparing test data")
    test_data = prepare_data_for_split(test, ir_embed)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Use optuna to maximize accuracy / parameters
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    trial = study.best_trial
    best_accuracy = trial.value
    best_precision = trial.user_attrs.get("precision")
    best_recall = trial.user_attrs.get("recall")
    best_f1_score = trial.user_attrs.get("f1_score")
    logger.info(
        f"Best trial completed with accuracy {trial.value:.4f}, precision {best_precision:.4f}, recall {best_recall:.4f}, and F1-Score {best_f1_score:.4f}."
    )

    # Recover best model
    model = GCN(
        in_channels=train_data[0].num_features,
        out_channels=2,
        hidden_channels=trial.params.get("hidden_channels"),
        num_layers=trial.params.get("num_layers"),
        dropout=trial.params.get("dropout"),
    ).to(device)

    model.load_state_dict(trial.user_attrs.get("state"))

    model_dir = pathlib.Path(PARENT / f"./models/{cwe_id.upper()}")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = pathlib.Path(model_dir / f"{cwe_id.upper()}.model")
    param_path = pathlib.Path(model_dir / f"{cwe_id.upper()}.json")

    logger.info(f"Saving model to {model_path}")
    torch.save(model.state_dict(), model_path)

    logger.info(f"Saving params to {param_path}")
    with open(param_path, "w") as f:
        json.dump(trial.params, f)
