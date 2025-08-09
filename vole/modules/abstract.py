from angr import Project
from abc import ABC, abstractmethod
from angr.analyses.cfg import CFGFast
from torch_geometric.data import Data


class Module(ABC):
    """
    Declares properties and methods for executing a module
    """

    def __init__(self):
        self.project = None
        self.cfg = None
        self.embeds = None

    @property
    @abstractmethod
    def cwe_id(self):
        pass

    @property
    @abstractmethod
    def cwe_name(self):
        pass

    @property
    @abstractmethod
    def description(self):
        pass

    def set_project(self, project: Project):
        self.project = project

    def set_cfg(self, cfg: CFGFast):
        self.cfg = cfg

    def set_embeds(self, embeds: dict[int, Data]):
        self.embeds = embeds

    @abstractmethod
    def execute(self) -> tuple[int, str]:
        pass
