import numpy as np

from economics.economic_loss_model import EconomicLossModel


class PigouKnowledgeLossModel(EconomicLossModel):
    def __init__(self, data, c_s=1.0):
        """
        Modelo Pigouviano de pérdida de conocimiento por externalidades negativas.

        Parameters:
            data: dict con al menos 'graph', 'db', 'non_reproducible_refs'
            c_s: coste social por cada cita a un artículo no reproducible
        """
        super().__init__(data)
        self.graph = data["graph"]
        self.db = data["db"]
        self.nr = data["non_reproducible_refs"]
        self.c_s = c_s
        self.avg_citations = None
        self.loss = None

    def compute_loss(self):
        affected_nodes = self.db.get_all_affected_papers()
        citation_counts = [len(list(self.graph.successors(n))) for n in affected_nodes if n in self.graph]
        self.avg_citations = np.mean(citation_counts) if citation_counts else 0
        self.loss = self.nr * self.avg_citations * self.c_s
        return self.loss

    def description(self):
        return (
            f"Pigou-based Externality Model:\n"
            f"Loss = N_NR * avg_citations * C_s = {self.nr} * {self.avg_citations:.2f} * {self.c_s}\n"
            f"(Modelo de propagación del daño por citación de artículos no reproducibles)"
        )
