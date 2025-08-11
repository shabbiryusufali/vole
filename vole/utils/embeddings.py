import re
import sys
import string
import torch
import pathlib
import fasttext
import numpy as np

from .cfg import get_sub_cfgs
from .graph import to_torch_data, insert_node_attributes
from .vexir2vec.embeddings import SymbolicEmbeddings
from .vexir2vec.normalize import Normalizer

import utils.vexir2vec.model_OTA  # NOTE: MUST be imported this way

from angr import Project
from angr.analyses.cfg import CFGFast
from angr.knowledge_plugins.functions import Function
from collections import Counter
from pyvex.stmt import (
    IRStmt,
    AbiHint,
    IMark,
    NoOp,
    LoadG,
    Store,
    StoreG,
    Put,
    PutI,
    WrTmp,
    Dirty,
    Exit,
)
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data


PARENT = pathlib.Path(__file__).parent.resolve()
NODE_FEATS = [
    "ret",
    "refs",
    "call",
    "exit",
    "load",
    "dirty",
    "store",
    "boring",
    "syscall",
    "constants",
    "operations",
    "successors",
    "statements",
    "expressions",
    "instructions",
    "predecessors",
]


class IREmbeddings:
    def __init__(self, device, train: bool = False):
        self.train = train
        self.vocab = pathlib.Path(PARENT / "./vexir2vec/vocabulary.txt")
        self.model = pathlib.Path(PARENT / "../models/vexir2vec.model")
        self.embeddings = EmbeddingsWrapper(self.vocab)
        self.device = device

        # Patch the resolution of the model's source at runtime
        sys.modules["model_OTA"] = utils.vexir2vec.model_OTA

        self.vexir2vec = torch.load(
            self.model, map_location=self.device, weights_only=False
        )
        self.vexir2vec.eval()

    def get_function_embeddings(
        self, proj: Project, cfg: CFGFast
    ) -> dict[int, Data]:
        """
        Returns a dictionary of address-embedding pairs
        NOTE: Adapted from VexIR2Vec/embeddings/vexNet/embeddings.py/processFunc()
        """
        funcs = [func for func in cfg.functions.values()]
        extern_funcs = self.embeddings.process_extern_calls(funcs, proj)

        func_embeds = {}

        for func, sub_cfg in get_sub_cfgs(cfg):
            # In training, we only want the labelled functions
            if bool(
                self.train and 
                "good" not in func.name and
                "bad" not in func.name
            ):    
                continue

            str_refs = self.embeddings.process_string_refs(func, False)

            blocks = {block.addr: block for block in func.blocks}
            block_walks = self.embeddings.randomWalk(func, blocks.keys())

            # Binary classification (1 == "bad", 0 == "good")
            # NOTE: Assume bad if proper label can't be deduced
            label = 1 if "bad" in func.name else 0 if "good" in func.name else 1

            func_opc_vec = np.zeros(self.embeddings.dim, dtype=np.float32)
            func_ty_vec = np.zeros(self.embeddings.dim, dtype=np.float32)
            func_arg_vec = np.zeros(self.embeddings.dim, dtype=np.float32)
            func_str_vec = self.embeddings.get_str_emb(str_refs)
            func_lib_vec = self.embeddings.get_ext_lib_emb(
                extern_funcs.get(func.addr)
            ).reshape(-1)

            for block_walk in block_walks:
                for blocknode in block_walk:
                    block = blocks.get(blocknode.addr)

                    block_opc_vec = np.zeros(
                        self.embeddings.dim, dtype=np.float32
                    )
                    block_ty_vec = np.zeros(
                        self.embeddings.dim, dtype=np.float32
                    )
                    block_arg_vec = np.zeros(
                        self.embeddings.dim, dtype=np.float32
                    )

                    if not block:
                        continue

                    if not block.vex:
                        continue

                    if not block.vex.has_statements:
                        continue

                    inst_list = [
                        self.embeddings.replace_instructions_with_keywords(stmt)
                        for stmt in block.vex.statements
                        if stmt and not isinstance(stmt, (AbiHint, IMark, NoOp))
                    ]
                    norm_list = self.embeddings.normalize_instructions(
                        inst_list
                    )
                    block_opc_vec, block_ty_vec, block_arg_vec = (
                        self.embeddings.get_vector_triplets(norm_list)
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

            ir_tens = []

            for data in dataloader:
                with torch.no_grad():
                    opc_emb = data[0].to(self.device)
                    ty_emb = data[1].to(self.device)
                    arg_emb = data[2].to(self.device)
                    str_emb = data[3].to(self.device)
                    lib_emb = data[4].to(self.device)

                    # NOTE: [1, 100]
                    res, _ = self.vexir2vec(
                        opc_emb, ty_emb, arg_emb, str_emb, lib_emb
                    )
                    # NOTE: [100]
                    ir_tens.append(res.squeeze(0))

            # NOTE: [1, 100]
            ir_tens = torch.stack(ir_tens)

            # Encode per-node information for some variance
            for node in sub_cfg.nodes():
                counter = Counter()
                cfg_node = cfg.model.get_any_node(node.addr)

                if cfg_node:
                    counter["successors"] += len(cfg_node.successors)
                    counter["predecessors"] += len(cfg_node.predecessors)
                    counter["refs"] += len(list(cfg_node.get_data_references()))

                    block = cfg_node.block

                    if block:
                        irsb = block.vex

                        if irsb:
                            counter["expressions"] += len(
                                list(irsb.expressions)
                            )
                            counter["instructions"] += irsb.instructions
                            counter["operations"] += len(irsb.operations)
                            counter["constants"] += len(irsb.constants)

                            if irsb.jumpkind == "Ijk_Call":
                                counter["call"] += 1

                            if irsb.jumpkind == "Ijk_Ret":
                                counter["ret"] += 1

                            if irsb.jumpkind == "Ijk_Boring":
                                counter["boring"] += 1

                            if irsb.jumpkind.startswith("Ijk_Sys"):
                                counter["syscall"] += 1

                            if irsb.has_statements:
                                for stmt in irsb.statements:
                                    counter["statements"] += 1

                                    if isinstance(stmt, (LoadG)):
                                        counter["load"] += 1

                                    if isinstance(
                                        stmt, (Store, StoreG, Put, PutI, WrTmp)
                                    ):
                                        counter["store"] += 1

                                    if isinstance(stmt, (Dirty)):
                                        counter["dirty"] += 1

                                    if isinstance(stmt, (Exit)):
                                        counter["exit"] += 1

                node_vec = np.array(
                    [counter.get(feat, 0) for feat in NODE_FEATS],
                    dtype=np.float32,
                )
                node_tens = torch.tensor(node_vec).to(self.device)
                node_tens = node_tens.unsqueeze(0)

                # NOTE: [1, 116]
                combined = torch.cat([ir_tens, node_tens], dim=1)

                # NOTE: [116]
                combined = combined.squeeze(0)

                # Insert features as node attributes
                # This ensures the values are preserved by torch later
                insert_node_attributes(sub_cfg, {node: {"y": label}})
                insert_node_attributes(sub_cfg, {node: {"x": combined}})

            func_embeds[func.addr] = to_torch_data(sub_cfg)

        return func_embeds


class EmbeddingsWrapper(SymbolicEmbeddings):
    """
    Wraps `SymbolicEmbeddings` to provide additional functionality
    NOTE: The authors suggest normalization level 3 unless otherwise necessary
    TODO: But the default value is 1 ...?
    """

    def __init__(self, vocab: pathlib.Path, normalize: int = 3):
        super().__init__(vocab)
        self.norm = Normalizer(normalize)
        self.model = pathlib.Path(PARENT / "../models/cc.en.100.bin")
        self.ft = fasttext.load_model(str(self.model))

    def replace_instructions_with_keywords(
        self, stmt: IRStmt
    ) -> list[str] | None:
        """
        NOTE: Adapted from VexIR2Vec/embeddings/vexNet/embeddings.py/processFunc()
        """
        stmt_str = str(stmt).lower()

        if stmt.tag == "Ist_Dirty":
            return

        tokens = self.tokenize(stmt_str)

        func_flag = False
        puti_flag = False
        geti_flag = False

        if stmt.tag == "Ist_WrTmp":
            if stmt.data.tag == "Iex_CCall":
                func_flag = True
            elif stmt.data.tag == "Iex_GetI":
                geti_flag = True

        if stmt.tag == "Ist_PutI":
            puti_flag = True

        new_stmt = stmt_str

        for i, token in enumerate(tokens):
            replace_token = token
            try:
                if i == 1:
                    if func_flag is True:
                        replace_token = "function"
                    elif puti_flag is True:
                        replace_token = "r1234"
                elif i == 2:
                    if geti_flag is True:
                        replace_token = "r1234"
                    elif puti_flag is True:
                        replace_token = "remove"
                elif i == 3 and geti_flag is True:
                    replace_token = "remove"

                new_stmt = new_stmt.replace(
                    token, self.keywords.getKeyword(replace_token), 1
                )

            except KeyError:
                # VARIABLE, CONSTANT, REGISTER, INTEGER, FLOAT, VECTOR, DECIMAL are not in keywords dict; so allow them
                pass

        return new_stmt

    def normalize_instructions(self, inst_list: list[str]) -> list[str]:
        return self.norm.transformWindow(inst_list)

    def get_vector_triplets(
        self, norm_list: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        NOTE: Adapted from VexIR2Vec/embeddings/vexNet/embeddings.py/processFunc()
        """
        block_opc_vec = np.zeros(self.dim, dtype=np.float32)
        block_ty_vec = np.zeros(self.dim, dtype=np.float32)
        block_arg_vec = np.zeros(self.dim, dtype=np.float32)

        for norm_stmt in norm_list:
            if norm_stmt is None or norm_stmt == "":
                continue

            stmt_opc_vec = np.zeros(self.dim, dtype=np.float32)
            stmt_ty_vec = np.zeros(self.dim, dtype=np.float32)
            stmt_arg_vec = np.zeros(self.dim, dtype=np.float32)

            tokens = self.tokenize(norm_stmt)
            tokens = self.canonicalizeTokens(tokens)

            for token in tokens:
                val_opc = np.zeros(self.dim, dtype=np.float32)
                val_ty = np.zeros(self.dim, dtype=np.float32)
                val_arg = np.zeros(self.dim, dtype=np.float32)

                if token in [
                    "register",
                    "integer",
                    "float",
                    "vector",
                    "decimal",
                ] or token.startswith("ity_"):
                    val_ty = self._lookupVocab(token)
                elif token in [
                    "variable",
                    "constant",
                    "jump_kind",
                    "none",
                    "dirty",
                    "exit",
                ] or token.startswith(
                    (
                        "ilgop_",
                        "imbe_",
                        "catoddlanesv",
                        "catevenlanesv",
                        "ctz",
                        "qnarrowbin",
                        "clz",
                    )
                ):
                    val_arg = self._lookupVocab(token)
                else:
                    val_opc = self._lookupVocab(token)

                stmt_opc_vec += val_opc
                stmt_ty_vec += val_ty
                stmt_arg_vec += val_arg

            block_opc_vec += stmt_opc_vec
            block_ty_vec += stmt_ty_vec
            block_arg_vec += stmt_arg_vec

        return (block_opc_vec, block_ty_vec, block_arg_vec)

    def get_embedding(self, str_refs: list) -> np.ndarray:
        vectors = [
            self.ft.get_word_vector(word).reshape((1, 100)) for word in str_refs
        ]
        return np.sum(np.concatenate(vectors, axis=0), axis=0, dtype=np.float32)

    def get_ext_lib_emb(self, ext_libs: list) -> np.ndarray:
        lib_vec = (
            self.get_embedding(ext_libs)
            if ext_libs
            else np.zeros(self.dim, dtype=np.float32)
        )
        return lib_vec.reshape((1, -1))

    def get_str_emb(self, str_refs: list[str]) -> np.ndarray:
        return (
            self.get_embedding(str_refs)
            if str_refs
            else np.zeros(self.dim, dtype=np.float32)
        )

    def remove_entity(self, text, entities: list):
        for entity in entities:
            text = text.replace(entity, " ")
        return text.strip()

    def process_string_refs(self, func, isUnstripped) -> list[str]:
        str_refs = []

        for _, str_ref in func.string_references():
            if isUnstripped:
                break
            """
            preprocess stringrefs for cleaning
            1. removing everything other than alphabets
            2. removing strings containing paths
            3. removing format specifiers
            4. lowercasing everything
            5. convert to separate tokens
            """
            # print("Debug str_ref: ",str_ref)
            str_ref = str_ref.decode("latin-1")
            if "http" in str_ref:
                continue
            format_specifiers = [
                "%c",
                "%s",
                "%hi",
                "%h",
                "%Lf",
                "%n",
                "%d",
                "%i",
                "%o",
                "%x",
                "%p",
                "%f",
                "%u",
                "%e",
                "%E",
                "%%",
                "%#lx",
                "%lu",
                "%ld",
                "__",
                "_",
            ]
            punctuations = list(string.punctuation)
            str_ref = self.remove_entity(str_ref, format_specifiers)
            str_ref = self.remove_entity(str_ref, punctuations)
            str_ref = re.sub("[^a-zA-Z]", " ", str_ref).lower().strip().split()
            if str_ref:
                str_refs.extend(str_ref)

        return str_refs

    def process_extern_calls(
        self, funcs: list[Function], project: Project
    ) -> dict[int, list]:
        """
        NOTE: Adapted from VexIR2Vex/vexNet/utils.py
        """
        ext_lib_funcs = [
            s.name
            for s in project.loader.symbols
            if s.is_function and s.is_extern
        ]
        ext_lib_fn_names = {}

        for func in funcs:
            call_sites = func.get_call_sites()

            for call_site in call_sites:
                callee = project.kb.functions.function(
                    func.get_call_target(call_site)
                )

                if callee is None:
                    continue
                if callee.name in ext_lib_funcs:
                    if func.addr in ext_lib_fn_names:
                        ext_lib_fn_names[func.addr].append(callee.name)
                    else:
                        ext_lib_fn_names[func.addr] = [callee.name]

        return ext_lib_fn_names
