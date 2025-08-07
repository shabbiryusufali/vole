import re
import string
import pathlib
import fasttext
import numpy as np

from .vexir2vec.embeddings import SymbolicEmbeddings
from .vexir2vec.normalize import Normalizer

from pyvex.stmt import IRStmt
from angr import Project
from angr.knowledge_plugins.functions import Function


HERE = pathlib.Path(__file__).parent.resolve()
MODEL = pathlib.Path(HERE / "../models/cc.en.100.bin")


class EmbeddingsWrapper(SymbolicEmbeddings):
    """
    Wraps `SymbolicEmbeddings` to provide additional functionality
    NOTE: The authors suggest normalization level 3 unless otherwise necessary
    TODO: But the default value is 1 ...?
    """

    def __init__(self, vocab: pathlib.Path, normalize: int = 3):
        super().__init__(vocab)
        self.norm = Normalizer(normalize)
        self.ft = fasttext.load_model(str(MODEL))

    def replace_instructions_with_keywords(self, stmt: IRStmt) -> list[str]:
        """
        NOTE: Adapted from VexIR2Vec/embeddings/vexNet/embeddings.py/processFunc()
        """
        stmt_str = str(stmt).lower()
        tokens = self.tokenize(stmt_str)

        if stmt.tag == "Ist_Dirty":
            # TODO: HMMM
            pass

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
            stmt_opc_vec = np.zeros(self.dim, dtype=np.float32)
            stmt_ty_vec = np.zeros(self.dim, dtype=np.float32)
            stmt_arg_vec = np.zeros(self.dim, dtype=np.float32)
            if norm_stmt is None or norm_stmt == "":
                continue

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
            if entity in text:
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
