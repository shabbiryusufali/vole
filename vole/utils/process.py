import re
import string
import fasttext
import fasttext.util
import numpy as np


class StringUtils:
    """
    NOTE: Adapted from VexIR2Vex/vexNet/utils.py
    """

    def __init__(self):
        fasttext.util.download_model("en", if_exists="ignore")
        self.ft = fasttext.load_model("cc.en.300.bin")
        fasttext.util.reduce_model(self.ft, 100)

    def get_embedding(self, str_refs):
        vectors = [
            self.ft.get_word_vector(word).reshape((1, 100)) for word in str_refs
        ]
        return np.sum(np.concatenate(vectors, axis=0), axis=0)

    def get_str_emb(self, str_refs: list[str]):
        return self.get_embedding(str_refs) if str_refs else np.zeros(100)

    def remove_entity(self, text, entity_list):
        for entity in entity_list:
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


def process_extern_calls(funcs, project) -> tuple[dict, dict, dict, set]:
    """
    NOTE: Adapted from VexIR2Vex/vexNet/utils.py
    """
    func_addr_list = [func.addr for func in funcs]
    ext_lib_funcs = [
        s.name for s in project.loader.symbols if s.is_function and s.is_extern
    ]
    ext_lib_fn_names = {}
    extern_edges = {}
    edges = {}
    called_set = set()

    for func in funcs:
        call_sites = func.get_call_sites()
        callee_func_list = [
            project.kb.functions.function(func.get_call_target(call_site))
            for call_site in call_sites
        ]
        extern_addr_list = []
        callee_addr_list = []
        for callee in callee_func_list:
            if callee is None:
                continue
            if callee.name in ext_lib_funcs:
                if func.addr in ext_lib_fn_names:
                    ext_lib_fn_names[func.addr].append(callee.name)
                else:
                    ext_lib_fn_names[func.addr] = [callee.name]
            if callee.addr not in func_addr_list:
                extern_addr_list.append(callee.addr)
            elif callee.addr != func.addr:
                callee_addr_list.append(callee.addr)
        extern_edges[func.addr] = extern_addr_list
        edges[func.addr] = callee_addr_list
        if callee_addr_list:
            called_set.update(callee_addr_list)

    return ext_lib_fn_names, extern_edges, edges, called_set
