import shutil
import pathlib
import argparse
import subprocess


COMPILERS = [
    ("gcc-10", "g++-10"),
    ("gcc-11", "g++-11"),
    ("gcc-12", "g++-12"),
    ("clang-12", "clang++-12"),
    ("clang-13", "clang++-13"),
    ("clang-14", "clang++-14"),
]
OPTS = ["O0", "O1", "O2"]


def make(cwe_id: str, path: str) -> None:
    """
    Makes the test cases in the subdirectory of `path` corresponding to `cwe_id`
    """
    cwe_id = cwe_id.upper()
    path = pathlib.Path(path)
    binary_pattern = f"{cwe_id}*.o"
    makefile_pattern = f"{cwe_id}*/**/Makefile"

    for makefile in path.rglob(makefile_pattern):
        parent = makefile.parent

        for cc, cpp in COMPILERS:
            # Create subdir for each compiler version used
            cc_dir = pathlib.Path(parent / f"{cc}_{cpp}")
            cc_dir.mkdir(parents=True, exist_ok=True)

            for opt in OPTS:
                subprocess.run(
                    [
                        "make",
                        "-C",
                        parent,
                        "-O",
                        f"CC={cc}",
                        f"CPP={cpp}",
                        f"CFLAGS=-c -{opt}",
                    ]
                )

                # Create nested subfolder for each optimization used
                opt_dir = pathlib.Path(cc_dir / f"{opt}")
                opt_dir.mkdir(parents=True, exist_ok=True)

                # Move binaries to their respective subfolders
                for binary in parent.glob(binary_pattern):
                    shutil.move(binary, opt_dir / binary.name)


def clean(cwe_id: str, path: str) -> None:
    """
    Invokes `make clean` in the subdirectory of `path` corresponding to `cwe_id`
    """
    cwe_id = cwe_id.upper()
    path = pathlib.Path(path)
    makefile_pattern = f"{cwe_id}*/**/Makefile"

    for makefile in path.rglob(makefile_pattern):
        parent = makefile.parent
        subprocess.run(["make", "clean", "-C", parent])

        for cc, cpp in COMPILERS:
            cc_dir = pathlib.Path(parent / f"{cc}_{cpp}")
            shutil.rmtree(cc_dir)


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
