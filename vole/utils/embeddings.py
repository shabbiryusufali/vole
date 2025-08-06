import pathlib
import numpy as np

from .vexir2vec.embeddings import SymbolicEmbeddings
from .vexir2vec.normalize import Normalizer

from pyvex.stmt import IRStmt


class EmbeddingWrapper(SymbolicEmbeddings):
    """
    Wraps `SymbolicEmbeddings` to provide additional functionality
    NOTE: The authors suggest normalization level 3 unless otherwise necessary
    TODO: But the default value is 1 ...?
    """

    def __init__(self, vocab: pathlib.Path, normalize: int = 3):
        super().__init__(vocab)
        self.norm = Normalizer(normalize)

    def replace_instructions_with_keywords(
        self, stmts: list[IRStmt] | None
    ) -> list[str]:
        """
        NOTE: Adapted from VexIR2Vec/embeddings/vexNet/embeddings.py/processFunc()
        """
        if stmts is None:
            return []

        inst_list = []
        for stmt in stmts:
            stmt_str = str(stmt).lower()
            tokens = self.tokenize(stmt_str)

            if stmt.tag == "Ist_Dirty":
                continue

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

            inst_list.append(new_stmt)

        return inst_list

    def normalize_instructions(self, inst_list: list[str]) -> list[str]:
        return self.norm.transformWindow(inst_list)

    def get_vector_triplets(
        self, norm_list: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        NOTE: Adapted from VexIR2Vec/embeddings/vexNet/embeddings.py/processFunc()
        """
        block_O_Vec = np.zeros(self.dim)
        block_T_Vec = np.zeros(self.dim)
        block_A_Vec = np.zeros(self.dim)

        for norm_stmt in norm_list:
            stmt_O_Vec = np.zeros(self.dim)
            stmt_T_Vec = np.zeros(self.dim)
            stmt_A_Vec = np.zeros(self.dim)
            if norm_stmt is None or norm_stmt == "":
                continue

            tokens = self.tokenize(norm_stmt)
            tokens = self.canonicalizeTokens(tokens)

            for token in tokens:
                val_O = np.zeros(self.dim)
                val_T = np.zeros(self.dim)
                val_A = np.zeros(self.dim)
                if token in [
                    "register",
                    "integer",
                    "float",
                    "vector",
                    "decimal",
                ] or token.startswith("ity_"):
                    val_T = self._lookupVocab(token)
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
                    val_A = self._lookupVocab(token)
                else:
                    val_O = self._lookupVocab(token)

                stmt_O_Vec += val_O
                stmt_T_Vec += val_T
                stmt_A_Vec += val_A

            block_O_Vec += stmt_O_Vec
            block_T_Vec += stmt_T_Vec
            block_A_Vec += stmt_A_Vec

        return (block_O_Vec, block_T_Vec, block_A_Vec)
