import json
import torch
import pathlib

from .abstract import Module

from angr.knowledge_plugins.functions import Function
from torch_geometric.nn.models import GCN

PARENT = pathlib.Path(__file__).parent.resolve()


class CWE416(Module):
    def __init__(self, project, cfg, device, embeddings):
        super().__init__(project, cfg, device, embeddings)
        self.param_path = pathlib.Path(PARENT / "../models/CWE416/CWE416.json")
        self.model_path = pathlib.Path(PARENT / "../models/CWE416/CWE416.model")
        
        with open(self.param_path, "r") as f:
            self.params = json.load(f)

        self.model = GCN(
            in_channels=116,
            out_channels=2,
            hidden_channels=self.params.get("hidden_channels"),
            num_layers=self.params.get("num_layers"),
            dropout=self.params.get("dropout")
        )
        self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

    @property
    def cwe_id(self):
        return "CWE416"

    @property
    def cwe_name(self):
        return "Use After Free"

    @property
    def description(self):
        return "Object is reused or referenced after it has been freed."

    def execute(self) -> tuple[dict, list[str]] | None:
        for addr, embed in self.embeds.items():
            embed.to(self.device)
            out = self.model(embed.x, embed.edge_index)
            pred = out.argmax(dim=1)

            func = self.cfg.kb.functions.get(addr)
            nodes = func.transition_graph.nodes()

            if len(nodes) > 0:
                pairs = {k: v for k, v in zip(nodes, pred) if v == 1}
                
                warns = []
                for node in nodes:
                    if isinstance(node, Function):
                        warns.append(self.warn(node, node.addr))
                    else:
                        warns.append(self.warn(func, node.addr))

                return pairs, warns
