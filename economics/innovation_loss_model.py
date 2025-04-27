import numpy as np
import pandas as pd

from .base_innovation_loss_model import BaseInnovationLossModel


class InnovationLossModel(BaseInnovationLossModel):
    def __init__(self, data, p, top_percentile=0.95):
        """
        Modelo con cálculo de I_avg mediante softmax ponderado por slowdown dinámico.

        data: dict con 'graph', 'db', 'non_reproducible_refs'
        p: probabilidad de innovación
        top_percentile: umbral para definir nodos disruptivos
        """
        super().__init__(data, p)
        self.graph = data["graph"]
        self.db = data["db"]
        self.nr = data["non_reproducible_refs"]
        self.p = p
        self.top_percentile = top_percentile
        self.i_avg = None
        self.df = None

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def _compute_i_avg_from_softmax(self):
        graph = self.graph
        db = self.db
        nodes = list(graph.nodes())
        citation_counts = {n: len(list(graph.successors(n))) for n in nodes}
        threshold = np.percentile(list(citation_counts.values()), self.top_percentile * 100)

        scores = []
        slowdown_factors = []

        for i, node in enumerate(nodes):
            print(f"Processing node {i + 1}/{len(nodes)}: {node}")
            is_affected = db.is_paper_affected(node)
            citations = citation_counts[node]

            if is_affected:
                slowdown = 0.3
            elif citations >= threshold:
                slowdown = 1.2
            else:
                slowdown = 1.0

            slowdown_factors.append(slowdown)
            scores.append(citations * slowdown)

        weights = self._softmax(np.array(scores))
        i_avg_total = sum(weights * len(nodes))  # innovación total ponderada
        self.i_avg = i_avg_total / self.nr if self.nr > 0 else 0

        self.df = pd.DataFrame({
            "node": nodes,
            "citations": [citation_counts[n] for n in nodes],
            "slowdown": slowdown_factors,
            "score": scores,
            "softmax_weight": weights
        })

        self._annotate_graph()

    def _annotate_graph(self):
        for _, row in self.df.iterrows():
            node = row["node"]
            if "innovation" not in self.graph.nodes[node]:
                self.graph.nodes[node]["innovation"] = {}
            self.graph.nodes[node]["innovation"]["i_avg_softmax"] = {
                "citations": row["citations"],
                "slowdown": row["slowdown"],
                "softmax_weight": row["softmax_weight"]
            }

    def compute_loss(self):
        self._compute_i_avg_from_softmax()
        return self.nr * self.i_avg * self.p

    def description(self):
        return (
            f"Innovation Loss (softmax-based I_avg): "
            f"Loss = N_R * I_avg * P = {self.nr} * {self.i_avg:.4f} * {self.p}\n"
            f"(I_avg obtenido mediante softmax con slowdown dinámico por nodo)"
        )
