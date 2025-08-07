import argparse
import pathlib
import subprocess


def make(cwe_id: str, path: str) -> None:
    """
    Makes the test cases in the subdirectory of `path` corresponding to `cwe_id`
    """
    cwe_id = cwe_id.upper()
    path = pathlib.Path(path)
    pattern = f"{cwe_id}*/**/Makefile"

    for makedir in path.rglob(pattern):
        subprocess.run(["make", "-C", makedir.parent])


def clean(cwe_id: str, path: str) -> None:
    """
    Invokes `make clean` in the subdirectory of `path` corresponding to `cwe_id`
    """
    cwe_id = cwe_id.upper()
    path = pathlib.Path(path)
    makefile_pattern = f"{cwe_id}*/**/Makefile"

    for makefile in path.rglob(makefile_pattern):
        subprocess.run(["make", "clean", "-C", makefile.parent])


def parse() -> dict:
    parser = argparse.ArgumentParser(
        prog="make.py",
        description="""
            Compiles SARD test cases
        """,
    )

    parser.add_argument("CWE-ID", type=str)
    parser.add_argument("path", type=pathlib.Path)
    parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        help="Whether or not to clean the compiled test cases",
    )

    return vars(parser.parse_args())


def main():
    args = parse()

    cwe_id = args.get("CWE-ID")
    path = args.get("path")
    should_clean = args.get("clean")

    if should_clean:
        clean(cwe_id, path)
    else:
        make(cwe_id, path)


if __name__ == "__main__":
    main()
