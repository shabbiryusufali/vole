import sys
import angr
import pathlib

from collections.abc import Generator


def crawl(cwe_id: str, path: str) -> Generator[pathlib.Path]:
    """
    Generator that yields binaries in `path` that match `cwe_id`
    """
    cwe_id = cwe_id.upper()
    path = pathlib.Path(path)
    pattern = f"**/{cwe_id}*.o"

    for file in path.rglob(pattern):
        yield file


def lift_blocks(func: angr.knowledge_plugins.functions.Function) -> str:
    """
    Lifts VEX IR from the basic blocks of a `func`
    """
    return "\n".join([str(block.vex) for block in func.blocks])


def lift(cwe_id: str, file: pathlib.Path) -> None:
    """
    Resolves external symbols from main binary and lifts corresponding VEX IR
    """
    proj = angr.Project(file, auto_load_libs=False)
    func_ir = {}  # (function name, IR) pairs

    # Find extern symbols matching `cwe_id` to lift
    for sym in proj.loader.main_object.imports:
        if sym.startswith(cwe_id.upper()):
            func_ir[sym] = None

    # Load each source binary
    for o in crawl(cwe_id, file.parent):
        temp_proj = angr.Project(o, auto_load_libs=False)
        temp_cfg = temp_proj.analyses.CFGFast()

        # Resolve symbols
        for sym in temp_proj.loader.main_object.symbols:
            if sym.is_function and sym.name in func_ir.keys():
                temp_func = temp_cfg.functions[sym.name]

                # Detect wrappers
                wrapper = False
                for call in temp_func.get_call_sites():
                    call_target = temp_func.get_call_target(call)
                    nested_func = temp_cfg.functions[call_target]

                    if nested_func.name in func_ir.keys():
                        continue

                    # TODO: Resolve translation error with certain nested functions
                    try:
                        if (
                            temp_func.name.endswith("good")
                            and "good" in nested_func.name
                        ):
                            name = f"{temp_func.name[:-4]}{nested_func.name}"
                            func_ir[name] = lift_blocks(nested_func)
                            wrapper = True

                        elif (
                            temp_func.name.endswith("bad") and "bad" in nested_func.name
                        ):
                            name = f"{temp_func.name[:-3]}{nested_func.name}"
                            func_ir[name] = lift_blocks(nested_func)
                            wrapper = True
                    except angr.errors.SimTranslationError:
                        continue

                if not wrapper:
                    func_ir[sym.name] = lift_blocks(temp_func)

    # Write IR to file
    for func, ir in func_ir.items():
        if ir:
            with open(f"{file.parent}/{func}.txt", "w") as f:
                f.write(ir)


def find_main(cwe_id: str, path: str) -> pathlib.Path:
    """
    Returns the path to `main_linux.o` in `path` that matches `cwe_id`
    """
    cwe_id = cwe_id.upper()
    path = pathlib.Path(path)
    pattern = f"{cwe_id}*/**/main_linux.o"
    main_dir = list(path.rglob(pattern))[0]
    return main_dir


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python rip.py [CWE-ID] [PATH]")
        sys.exit()

    main = find_main(sys.argv[1], sys.argv[2])
    lift(sys.argv[1], main)


if __name__ == "__main__":
    main()
