import argparse
import logging
import pathlib

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

defaults = {"output": "./output", "verbosity": "info"}


def parse() -> dict:
    parser = argparse.ArgumentParser(
        prog="vole.py",
        description="""
            Vulnerability Observance and Learning-based Exploitation (VOLE)
            is a tool for detecting important CWEs in program binaries
        """,
    )

    # Positional
    parser.add_argument("binary", type=pathlib.Path)

    # Options
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=defaults.get("output"),
        help="Where to output the results",
    )
    parser.add_argument(
        "-vvv",
        "--verbosity",
        type=str,
        default=defaults.get("verbosity"),
        choices=list(log_levels.keys()),
        help="How much log detail to output",
    )

    return vars(parser.parse_args())
