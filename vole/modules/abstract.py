from abc import ABC, abstractmethod


class Module(ABC):
    """
    Declares properties and methods for executing a module
    """

    def __init__(self, project, cfg, device, embeddings):
        self.project = project
        self.cfg = cfg
        self.device = device
        self.embeds = embeddings

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

    def warn(self, func, addr: int) -> str:
        return f"[{self.cwe_id}] ({self.cwe_name}) in {func.name} @ {hex(addr)}"

    @abstractmethod
    def execute(self) -> tuple[dict, str] | None:
        pass
