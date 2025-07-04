import argparse
import logging
import pathlib

from angr import Project
from modules import modules


logger = logging.getLogger(__name__)
log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


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
        default="./output",
        help="Where to output the results",
    )
    parser.add_argument(
        "-vvv",
        "--verbosity",
        type=str,
        default="info",
        choices=list(log_levels.keys()),
        help="How much log detail to output",
    )

    return vars(parser.parse_args())


def main():
    args = parse()

    # Configure logger with supplied verbosity level
    level = log_levels.get(args.get("verbosity"))
    logger.setLevel(level)

    # Configure project and generate CFG
    project = Project(args.get("binary"), auto_load_libs=False)
    cfg = project.analyses.CFGFast()
    warnings = []

    # Run each module and collect warnings
    for idx, module in enumerate(modules):
        logger.info(f"Running module {idx + 1}/{len(modules)}")

        module.set_project(project)
        module.set_CFG(cfg)

        warning = module.execute()
        warnings.append(warning)

    for warning in warnings:
        logger.warning(warning)


if __name__ == "__main__":
    main()
