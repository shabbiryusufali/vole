import sys
import math
import json
import torch
import optuna
import logging
import pathlib
import argparse
from datetime import datetime
from time import perf_counter

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
START = perf_counter()


def prepare_data_for_split(
    split: list[pathlib.Path], ir_embed: IREmbeddings
) -> list:
    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} +{perf_counter() - START:.1f}s Starting data preprocessing",
        flush=True,
    )

    split_data = []
    split_len = len(split)
    split_digits = int(math.log10(split_len)) + 1

    for idx, path in enumerate(split):
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} +{perf_counter() - START:.1f}s "
            f"[{str(idx + 1).rjust(split_digits)}/{split_len}] Processing path: {path}",
            flush=True,
        )

        proj, cfg = get_project_cfg(path)
        embeddings = ir_embed.get_function_embeddings(proj, cfg)
        logger.debug("extracted=%d from=%s", len(embeddings), path)
        split_data.extend(embeddings.values())

    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} +{perf_counter() - START:.1f}s Data preprocessing complete",
        flush=True,
    )

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


def do_testing(model: GCN) -> tuple[float]:
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
            predictions = pred.view(-1).long()
            actuals = batch.y.view(-1).long()
            
            true_positive = (predictions & actuals).sum().item()
            true_negative = ((predictions == 0) & (actuals == 0)).sum().item()
            false_positive = (predictions & (actuals == 0)).sum().item()
            false_negative = ((predictions == 0) & actuals).sum().item()
            
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
    return accuracy, precision, recall, f1_score


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

    logger.info(
        "trial=%d hidden=%d layers=%d dropout=%.4f lr=%.6f opt=%s",
        trial.number,
        hidden_channels,
        num_layers,
        dropout,
        lr,
        optimizer_name,
    )

    best_acc = 0.0
    for epoch in range(100):
        do_training(model, optimizer)
        accuracy, precision, recall, f1_score = do_testing(model)

        if accuracy > best_acc:
            best_acc = accuracy
        logger.debug(
            "trial=%d epoch=%d acc=%.4f best=%.4f",
            trial.number,
            epoch,
            accuracy,
            best_acc,
        )

        trial.report(accuracy, epoch)

        if trial.should_prune():
            logger.info(
                "trial=%d pruned_at_epoch=%d acc=%.4f",
                trial.number,
                epoch,
                accuracy,
            )
            raise optuna.exceptions.TrialPruned()

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
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} +{perf_counter() - START:.1f}s "
            f"""
            CWE-ID `{cwe_id}` and path `{path}` yielded no results.
            Check that `path` contains the compiled test cases.
            """,
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    # NOTE: `device` accessed above
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ir_embed = IREmbeddings(device, train=True)

    # NOTE: `train_data` and `train_loader` accessed above
    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} +{perf_counter() - START:.1f}s Preparing training data",
        flush=True,
    )
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
    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} +{perf_counter() - START:.1f}s Preparing test data",
        flush=True,
    )
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
    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} +{perf_counter() - START:.1f}s Best trial completed with accuracy {trial.value:.4f}, 
          precision {best_precision:.4f}, recall {best_recall:.4f}, and F1-Score {best_f1_score:.4f}.",
          flush=True,
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

    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} +{perf_counter() - START:.1f}s Saving model to {model_path}",
        flush=True,
    )
    torch.save(model.state_dict(), model_path)

    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} +{perf_counter() - START:.1f}s Saving params to {param_path}",
        flush=True,
    )
    with open(param_path, "w") as f:
        json.dump(trial.params, f)
