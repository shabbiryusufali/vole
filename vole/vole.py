import torch
import argparse
import logging
import pathlib

from modules import modules
from utils.cfg import get_project_cfg
from utils.embeddings import IREmbeddings


# Silence angr
logger = logging.getLogger("cle.backends.externs")
logger.setLevel(logging.ERROR)

logger = logging.getLogger("cle.loader")
logger.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


LOGLEVEL = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

DEFAULTS = {"output": "./output", "verbosity": "info"}


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
        default=DEFAULTS.get("output"),
        help="Where to output the results",
    )
    parser.add_argument(
        "-vvv",
        "--verbosity",
        type=str,
        default=DEFAULTS.get("verbosity"),
        choices=list(LOGLEVEL.keys()),
        help="How much log detail to output",
    )

    return vars(parser.parse_args())


def main():
    args = parse()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ir_embed = IREmbeddings(device)

    # Configure logger with supplied verbosity level
    level = LOGLEVEL.get(args.get("verbosity"))
    logger.setLevel(level)

    # Configure project and generate CFG
    proj, cfg = get_project_cfg(args.get("binary"))

    # Extract vector embeddings for each function
    embeds = ir_embed.get_function_embeddings(proj, cfg)

    warns = []
    vulns = {}

    # Run each module and collect warnings + interesting addresses
    for idx, module in enumerate(modules):
        logger.info(f"Running module {idx + 1}/{len(modules)}")

        module = module(project=proj, cfg=cfg, device=device, embeddings=embeds)

        res = module.execute()

        if res:
            vuln, warn = res
            vulns.update(vuln)
            warns.append(warn)


if __name__ == "__main__":
    main()
