import gensim
import pathlib
import numpy as np

from utils.io import crawl
from .cfg import lift_ir, get_program_cfg, get_sub_cfgs

from collections.abc import Iterator


def get_corpus_splits(cwe_id: str, path: pathlib.Path) -> tuple[list, list, list]:
    """
    Reads in paths of test files from `path` corresponding to `cwe_id`, shuffles them, and returns a 45-45-10 train-test-eval split
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


def read_corpus(split: list[pathlib.Path], tokens_only=False) -> Iterator:
    """
    Iterator that yields either `list[str]` or `gensim.models.doc2vec.TaggedDocument` depending on `tokens_only` for the given `split`
    REF: https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#define-a-function-to-read-and-preprocess-text
    """
    for path in split:
        cfg = get_program_cfg(path)
        for sub_cfg in get_sub_cfgs(cfg):
            for idx, (node, ir) in enumerate(lift_ir(sub_cfg)):
                tokens = gensim.utils.simple_preprocess(ir)
                if tokens_only:
                    yield tokens
                else:
                    yield gensim.models.doc2vec.TaggedDocument(tokens, [idx])
