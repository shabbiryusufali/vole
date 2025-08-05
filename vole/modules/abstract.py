from abc import ABC, abstractmethod


class Module(ABC):
    """
    Declares properties and methods for executing a module
    """

    def __init__(self):
        self.project = None

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

    def set_project(self, project):
        self.project = project

    @abstractmethod
    def execute(self) -> str:
        pass
