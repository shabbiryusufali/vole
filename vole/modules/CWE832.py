import json
import torch
import pathlib

from .abstract import Module

from torch_geometric.nn.models import GCN

PARENT = pathlib.Path(__file__).parent.resolve()


class CWE832(Module):
    def __init__(self, project, cfg, device, embeddings):
        super().__init__(project, cfg, device, embeddings)
        self.param_path = pathlib.Path(PARENT / "../models/CWE832/CWE832.json")
        self.model_path = pathlib.Path(PARENT / "../models/CWE832/CWE832.model")

        with open(self.param_path, "r") as f:
            self.params = json.load(f)

        self.model = GCN(
            in_channels=116,
            out_channels=2,
            hidden_channels=self.params.get("hidden_channels"),
            num_layers=self.params.get("num_layers"),
            dropout=self.params.get("dropout"),
        )
        self.model.load_state_dict(
            torch.load(self.model_path, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()

    @property
    def cwe_id(self):
        return "CWE832"

    @property
    def cwe_name(self):
        return "Unlock of a Resource that is not Locked"

    @property
    def description(self):
        return "Unlocking a resource that was never locked."
