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

    def warn(self, thing, addr: int) -> str:
        if hasattr(thing, "name"):
            return f"[{self.cwe_id}] ({self.cwe_name}) in {thing.name} @ {hex(addr)}"
        else:
            return f"[{self.cwe_id}] ({self.cwe_name}) @ {hex(addr)}"

    def execute(self) -> list[str]:
        warns = []
        for addr, embed in self.embeds.items():
            embed.to(self.device)
            out = self.model(embed.x, embed.edge_index)
            preds = out.argmax(dim=1)

            func = self.cfg.kb.functions.get(addr)
            nodes = func.transition_graph.nodes()

            if len(nodes) > 0:
                # NOTE: Not compiling to optimized byte code
                assert len(nodes) == len(preds)  # nosec B101

                for node, pred in zip(nodes, preds):
                    if pred == 1:
                        warns.append(self.warn(node, node.addr))

        return warns
