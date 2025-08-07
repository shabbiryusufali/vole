import sys
import torch
import logging
import pathlib
import argparse
import numpy as np

from utils.cfg import (
    get_project_cfg,
    get_sub_cfgs,
)
from utils.graph import (
    insert_node_attributes,
    to_torch_data,
)
from utils.train import get_corpus_splits
from utils.embeddings import EmbeddingsWrapper

import utils.vexir2vec.model_OTA  # NOTE: MUST be imported this way

from pyvex.stmt import AbiHint, IMark, IRStmt, NoOp
from torch.utils.data import TensorDataset, DataLoader

# Silence angr
logger = logging.getLogger("cle.backends.externs")
logger.setLevel(logging.ERROR)

logger = logging.getLogger("cle.loader")
logger.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TODO: Bad practice to define this at top level
HERE = pathlib.Path(__file__).parent.resolve()
VOCAB = pathlib.Path(HERE / "./utils/vexir2vec/vocabulary.txt")
MODEL = pathlib.Path(HERE / "./models/vexir2vec.model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDINGS = EmbeddingsWrapper(VOCAB)

# Patch the resolution of the model's source at runtime
sys.modules["model_OTA"] = utils.vexir2vec.model_OTA

# TODO: Figure out if it's possible to load the model with weights_only=True
vexir2vec = torch.load(MODEL, map_location=DEVICE, weights_only=False)
vexir2vec.eval()


def prepare_data_for_split(split: list[pathlib.Path]) -> list:
    split_data = []

    for path in split:
        proj, cfg = get_project_cfg(path)
        funcs = [func for func in cfg.functions.values()]
        extern_funcs = EMBEDDINGS.process_extern_calls(funcs, proj)

        for func, sub_cfg in get_sub_cfgs(cfg):
            str_refs = EMBEDDINGS.process_string_refs(func, False)

            blocks = {block.addr: block for block in func.blocks}
            block_walks = EMBEDDINGS.randomWalk(func, blocks.keys())

            # TODO: More granular labelling
            if not func.name:
                label = -1  # Invalid
            elif "bad" in func.name:
                label = 0  # Bad (i.e. vulnerable)
            elif "good" in func.name:
                label = 1  # Good
            else:
                label = -1

            func_opc_vec = np.zeros(EMBEDDINGS.dim, dtype=np.float32)
            func_ty_vec = np.zeros(EMBEDDINGS.dim, dtype=np.float32)
            func_arg_vec = np.zeros(EMBEDDINGS.dim, dtype=np.float32)
            func_str_vec = EMBEDDINGS.get_str_emb(str_refs)
            func_lib_vec = EMBEDDINGS.get_ext_lib_emb(
                extern_funcs.get(func.addr)
            ).reshape(-1)

            for block_walk in block_walks:
                for blocknode in block_walk:
                    block = blocks.get(blocknode.addr)

                    block_opc_vec = np.zeros(EMBEDDINGS.dim, dtype=np.float32)
                    block_ty_vec = np.zeros(EMBEDDINGS.dim, dtype=np.float32)
                    block_arg_vec = np.zeros(EMBEDDINGS.dim, dtype=np.float32)

                    if not block:
                        continue

                    if not block.vex:
                        continue

                    if not block.vex.has_statements:
                        continue

                    inst_list = [
                        EMBEDDINGS.replace_instructions_with_keywords(stmt)
                        for stmt in block.vex.statements
                        if not isinstance(stmt, (AbiHint, IMark, NoOp))
                    ]
                    norm_list = EMBEDDINGS.normalize_instructions(inst_list)
                    block_opc_vec, block_ty_vec, block_arg_vec = (
                        EMBEDDINGS.get_vector_triplets(norm_list)
                    )

                    func_opc_vec += block_opc_vec
                    func_ty_vec += block_ty_vec
                    func_arg_vec += block_arg_vec

            dataloader = DataLoader(
                TensorDataset(
                    torch.from_numpy(func_opc_vec).unsqueeze(0),
                    torch.from_numpy(func_ty_vec).unsqueeze(0),
                    torch.from_numpy(func_arg_vec).unsqueeze(0),
                    torch.from_numpy(func_str_vec).unsqueeze(0),
                    torch.from_numpy(func_lib_vec).unsqueeze(0),
                ),
                batch_size=1,
                shuffle=False,
                num_workers=12,
            )

            ir_vec = []
            for data in dataloader:
                with torch.no_grad():
                    opc_emb = data[0].to(DEVICE)
                    ty_emb = data[1].to(DEVICE)
                    arg_emb = data[2].to(DEVICE)
                    str_emb = data[3].to(DEVICE)
                    lib_emb = data[4].to(DEVICE)

                    res = vexir2vec(opc_emb, ty_emb, arg_emb, str_emb, lib_emb)
                    ir_vec.extend(res)

            for node in sub_cfg.nodes():
                # Insert features as node attributes
                # This ensures the values are preserved by torch later
                insert_node_attributes(sub_cfg, {node: {"label": label}})
                insert_node_attributes(sub_cfg, {node: {"ir_vec": ir_vec}})

            split_data.append(to_torch_data(sub_cfg))

    return split_data


def train_gcn(cwe_id: str, path: pathlib.Path):
    """
    Trains a `torch_geometric.nn.models.GCN` from SARD test case binaries
    matching `cwe_id` in `path`
    """
    train, test, evaluation = get_corpus_splits(cwe_id, path)

    if not all((train, test, evaluation)):
        logger.info(
            f"""
            CWE-ID `{cwe_id}` and path `{path}` yielded no results.
            Check that `path` contains the compiled test cases.
            """
        )

    prepare_data_for_split(train)

    # TODO: Train GCN on `training_data`


def parse() -> dict:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="""
            Train a PyTorch neural network on SARD test cases
        """,
    )

    parser.add_argument("CWE-ID", type=str)
    parser.add_argument("path", type=pathlib.Path)

    return vars(parser.parse_args())


def main():
    args = parse()
    train_gcn(args.get("CWE-ID"), args.get("path"))


if __name__ == "__main__":
    main()
