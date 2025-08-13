# Adding New Modules

To add a new module:

1. Create a new Python file in this directory
2. Within this file, define a class which inherits from `Module` (see `abstract.py`)
3. Perform all initialization in the `__init__` method
    1. Note: If overriding `__init__`, make sure to accept the parameters `project`, `cfg`, `device`, `embeddings`, and pass them to the superclass's constructor like so: `super().__init__(project, cfg, device, embeddings)`
4. Define the following properties on the class:
    1. `cwe_id`
    2. `cwe_name`
    3. `description`

For example, 

```py
import json
import torch
import pathlib

from .abstract import Module

from torch_geometric.nn.models import GCN

PARENT = pathlib.Path(__file__).parent.resolve()


class MyNewSuperCoolModule(Module):
    def __init__(self, project, cfg, device, embeddings):
        super().__init__(project, cfg, device, embeddings)
        self.param_path = pathlib.Path(PARENT / "../models/MyNewSuperCoolModule/MyNewSuperCoolModule.json")
        self.model_path = pathlib.Path(PARENT / "../models/MyNewSuperCoolModule/MyNewSuperCoolModule.model")

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
        return "<CWE-ID>"

    @property
    def cwe_name(self):
        return "<CWE-Name>"

    @property
    def description(self):
        return "<CWE-Description>"

```
