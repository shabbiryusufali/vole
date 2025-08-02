import sys
import angr
import logging
import pathlib

from collections.abc import Generator


# Silence, fools
# We know the symbol was allocated without a known size
# And we can't do anything about it!
logger = logging.getLogger("cle.backends.externs")
logger.setLevel(logging.ERROR)

logger = logging.getLogger("cle.loader")
logger.setLevel(logging.ERROR)


def crawl(path: pathlib.Path, pattern: str) -> Generator[pathlib.Path]:
    """
    Generator that yields `pathlib.Path`s in `path` that match `pattern`
    """
    for file in path.rglob(pattern):
        yield file


def lift_blocks(func: angr.knowledge_plugins.functions.Function) -> str:
    """
    Lifts VEX IR from the basic blocks of a `func` as a `str`
    """
    return "\n".join([str(block.vex) for block in func.blocks])


def lift(cwe_id: str, file: pathlib.Path) -> None:
    """
    Resolves external symbols from main binary and lifts corresponding VEX IR
    """

    # Find extern symbols matching `cwe_id` to lift
    project = angr.Project(file, auto_load_libs=False)
    imports = project.loader.main_object.imports
    symbols = [s for s in imports if cwe_id.upper() in s]

    # Load each source binary
    for o in crawl(file.parent, f"**/{cwe_id}*.o"):
        src_proj = angr.Project(o, auto_load_libs=False)
        src_cfg = src_proj.analyses.CFGFast()

        # Resolve symbols
        for sym in src_proj.loader.main_object.symbols:
            # Ignore non-functions
            if not sym.is_function:
                continue

            # Ignore functions not invoked by the main binary
            if sym.name not in symbols:
                continue

            src_func = src_cfg.functions.get(sym.name)
            has_nested_funcs = False

            # Resolve called functions (which may contain the code we want)
            for call in src_func.get_call_sites():
                call_target = src_func.get_call_target(call)
                nested_func = src_cfg.functions[call_target]

                if cwe_id.upper() in nested_func.name:
                    has_nested_funcs = True

                    try:
                        with open(
                            f"{file.parent}/{file.stem}_{nested_func.name}.txt", "w"
                        ) as f:
                            f.write(lift_blocks(nested_func))
                    except angr.errors.SimTranslationError:
                        print(f"Failed to translate {nested_func.name}")

            if not has_nested_funcs:
                with open(f"{file.parent}/{file.stem}_{src_func.name}.txt", "w") as f:
                    f.write(lift_blocks(src_func))


def main():
    if len(sys.argv) != 3:
        print("Usage: python rip_ir.py [CWE-ID] [PATH]")
        sys.exit()

    cwe_id = sys.argv[1]
    path = pathlib.Path(sys.argv[2])

    for m in crawl(path, f"{cwe_id}*/**/main_linux.o"):
        lift(cwe_id, m)


if __name__ == "__main__":
    main()
