import pathlib
import numpy as np

from utils.io import crawl


def get_corpus_splits(cwe_id: str, path: pathlib.Path) -> tuple[list, list, list]:
    """
    Reads in paths of test files from `path` corresponding to `cwe_id`, 
    shuffles them, and returns a 45-45-10 train-test-eval split
    """
    full = []
    for m in crawl(path, f"{cwe_id}*/**/main_linux.o"):
        for c in crawl(m.parent, f"**/{cwe_id}*.o"):
            full.append(c)

    full = np.array(full)
    np.random.shuffle(full)
    full = full.tolist()

    fourty_five = int(0.45 * len(full))
    return (
        full[:fourty_five],
        full[fourty_five : (2 * fourty_five)],
        full[(2 * fourty_five) :],
    )
