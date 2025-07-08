import sys
import pathlib
import subprocess


def make(cwe_id: str, path: str) -> None:
    """
    Makes the test cases in the subdirectory of `path` corresponding to `cwe_id`
    """
    cwe_id = cwe_id.upper()
    path = pathlib.Path(path)
    pattern = f"{cwe_id}*/**/Makefile"
    makedir = list(path.rglob(pattern))[0].parent
    subprocess.run(["make", "-C", makedir])


def clean(cwe_id: str, path: str) -> None:
    """
    Invokes `make clean` in the subdirectory of `path` corresponding to `cwe_id`
    Also cleans up `.txt` files created by `rip.py`
    """
    cwe_id = cwe_id.upper()
    path = pathlib.Path(path)
    pattern = f"{cwe_id}*/**/Makefile"
    makedir = list(path.rglob(pattern))[0].parent
    subprocess.run(["make", "clean", "-C", makedir])

    pattern = f"{cwe_id}*/**/{cwe_id}*.txt"
    for txt in path.rglob(pattern):
        txt.unlink(missing_ok=True)


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python make.py [CWE-ID] [PATH] {clean}")
        sys.exit()

    if len(sys.argv) == 3:
        make(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        if sys.argv[3] == "clean":
            clean(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
