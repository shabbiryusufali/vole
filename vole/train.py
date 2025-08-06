import sys
import torch
import logging
import pathlib
import argparse

from utils.cfg import (
    get_project_cfg,
    get_sub_cfgs,
    lift_stmt_ir,
)
from utils.graph import (
    get_digraph_source_nodes,
    insert_node_attributes,
    to_torch_data,
)
from utils.train import get_corpus_splits
from utils.embeddings import EmbeddingWrapper
from utils.process import StringUtils, process_extern_calls

import utils.vexir2vec.model_OTA  # NOTE: MUST be imported this way
from torch.utils.data import TensorDataset, DataLoader

# Silence angr
logger = logging.getLogger("cle.backends.externs")
logger.setLevel(logging.ERROR)

logger = logging.getLogger("cle.loader")
logger.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


HERE = pathlib.Path(__file__).parent.resolve()
VOCAB = pathlib.Path(
    HERE / "./utils/vexir2vec/vocabulary.txt"
)  # NOTE: Might be wrong path since the base class is in `vexir2vec`
MODEL = pathlib.Path(HERE / "./models/vexir2vec.model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRUTILS = StringUtils()

# Patch the resolution of the model's source at runtime
sys.modules["model_OTA"] = utils.vexir2vec.model_OTA

# TODO: Figure out if it's possible to load the model with weights_only=True
vexir2vec = torch.load(MODEL, map_location=DEVICE, weights_only=False)
vexir2vec.eval()


def prepare_data_for_split(split: list[pathlib.Path]) -> list:
    split_data = []
    embeddings = EmbeddingWrapper(VOCAB)

    for path in split:
        proj, cfg = get_project_cfg(path)
        funcs = [func for func in cfg.functions.values()]
        (
            ext_lib_fn_names,
            extern_edges,
            edges,
            called_set,
        ) = process_extern_calls(funcs, proj)

        for func, sub_cfg in get_sub_cfgs(cfg):
            source = get_digraph_source_nodes(sub_cfg)[0]
            name = sub_cfg.nodes[source].get("name")
            str_refs = STRUTILS.process_string_refs(func, False)
            block_str_vec = STRUTILS.get_str_emb(str_refs)

            # TODO: More granular labelling
            if not name:
                label = -1  # Invalid
            elif "bad" in name:
                label = 0  # Bad (i.e. vulnerable)
            elif "good" in name:
                label = 1  # Good
            else:
                label = -1

            for node, stmts_ir in lift_stmt_ir(sub_cfg):
                # Preprocess data (into VexIR2Vec's expected format)
                inst_list = embeddings.replace_instructions_with_keywords(
                    stmts_ir
                )
                norm_list = embeddings.normalize_instructions(inst_list)
                (
                    block_opc_vec,
                    block_ty_vec,
                    block_arg_vec,
                ) = embeddings.get_vector_triplets(norm_list)

                # TODO: Compute this from ext_lib_fn_names, extern_edges, edges, and called_set somehow
                block_lib_vec = []

                dataset = TensorDataset(
                    block_opc_vec,
                    block_ty_vec,
                    block_arg_vec,
                    block_str_vec,
                    block_lib_vec,
                )
                dataloader = DataLoader(
                    dataset, batch_size=1, shuffle=False, num_workers=20
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
