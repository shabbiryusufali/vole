from abc import ABC, abstractmethod


class Module(ABC):
    """
    Declares properties and methods for executing a module
    """

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

    @abstractmethod
    def set_project(self, project):
        pass

    @abstractmethod
    def set_CFG(self, project):
        pass
    
    @abstractmethod
    def execute(self) -> None:
        pass
