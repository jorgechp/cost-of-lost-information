from abc import ABC, abstractmethod
from .economic_loss_model import EconomicLossModel


class BaseInnovationLossModel(EconomicLossModel, ABC):
    def __init__(self, data, p):
        """
        Base para modelos de pérdida de innovación.

        Parameters:
            data: debe incluir 'graph', 'db', 'non_reproducible_refs'
            p: probabilidad de innovación
        """
        super().__init__(data)
        self.graph = data["graph"]
        self.db = data["db"]
        self.nr = data["non_reproducible_refs"]
        self.p = p
        self.i_avg = None
        self.df = None

    @abstractmethod
    def compute_loss(self):
        pass

    @abstractmethod
    def _annotate_graph(self):
        pass

    @abstractmethod
    def description(self):
        pass
