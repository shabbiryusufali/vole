import pathlib
import numpy as np

from collections.abc import Iterator


def crawl(path: pathlib.Path, pattern: str) -> Iterator[pathlib.Path]:
    """
    Iterator that yields `pathlib.Path`s in `path` that match `pattern`
    """
    for file in path.rglob(pattern):
        yield file


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
