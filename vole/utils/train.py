import torch
import pathlib
import numpy as np

from utils.io import crawl

from torch_geometric.nn.models import GCN


def get_corpus_splits(
    cwe_id: str, path: pathlib.Path
) -> tuple[list, list, list]:
    """
    Reads in paths of test files from `path` corresponding to `cwe_id`,
    shuffles them, and returns a 50-50 train-test split
    """
    full = []
    for m in crawl(path, f"{cwe_id}*/**/main_linux.o"):
        for c in crawl(m.parent, f"**/{cwe_id}*.o"):
            full.append(c)

    full = np.array(full)
    np.random.shuffle(full)
    full = full.tolist()

    half = len(full) // 2
    return full[:half], full[half:]


def save_model(model, out_path: pathlib.Path) -> None:
    torch.save(model.state_dict(), out_path)


def load_model(model, in_path: pathlib.Path) -> GCN:
    model.load_state_dict(torch.load(in_path, weights_only=True))
    return model
