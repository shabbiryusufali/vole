from abc import abstractmethod


class Module:
    """
    Declares a method for executing a module
    """

    @abstractmethod
    def execute(self) -> None:
        pass
