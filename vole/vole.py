import logging

from angr import Project

from cli import parse, log_levels
from modules import modules


logger = logging.getLogger(__name__)


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
