import pathlib

from collections.abc import Iterator


def crawl(path: pathlib.Path, pattern: str) -> Iterator[pathlib.Path]:
    """
    Iterator that yields `pathlib.Path`s in `path` that match `pattern`
    """
    for file in path.rglob(pattern):
        yield file
