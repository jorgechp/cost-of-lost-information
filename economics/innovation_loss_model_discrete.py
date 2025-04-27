import numpy as np
import pandas as pd

from .base_innovation_loss_model import BaseInnovationLossModel


class InnovationLossModelDiscrete(BaseInnovationLossModel):
    def __init__(self, data, p, slowdown_values=None, top_percentile=0.95):
        """
        Modelo con slowdown fijo por clase: sano, afectado, disruptivo.

        data: dict con 'graph', 'db', 'non_reproducible_refs'
        p: probabilidad de innovación
        slowdown_values: dict opcional con valores {'sano', 'afectado', 'disruptivo'}
        top_percentile: umbral para definir nodos disruptivos
        """
        super().__init__(data, p)
        self.graph = data["graph"]
        self.db = data["db"]
        self.nr = data["non_reproducible_refs"]
        self.p = p
        self.top_percentile = top_percentile
        self.slowdown_values = slowdown_values or {
            "sano": 1.0,
            "afectado": 0.3,
            "disruptivo": 1.2
        }
        self.i_avg = None
        self.df = None

    def compute_loss(self):
        graph = self.graph
        db = self.db
        nodes = list(graph.nodes())
        citation_counts = {n: len(list(graph.successors(n))) for n in nodes}
        threshold = np.percentile(list(citation_counts.values()), self.top_percentile * 100)

        node_slowdowns = []
        for i, node in enumerate(nodes):
            print(f"Processing node {i + 1}/{len(nodes)}: {node}")
            citations = citation_counts[node]
            is_affected = db.is_paper_affected(node)

            if is_affected:
                slowdown = self.slowdown_values["afectado"]
            elif citations >= threshold:
                slowdown = self.slowdown_values["disruptivo"]
            else:
                slowdown = self.slowdown_values["sano"]

            node_slowdowns.append(slowdown)

        self.i_avg = np.mean(node_slowdowns)
        self.df = pd.DataFrame({
            "node": nodes,
            "citations": [citation_counts[n] for n in nodes],
            "slowdown": node_slowdowns
        })

        self._annotate_graph()
        return self.nr * self.i_avg * self.p

    def _annotate_graph(self):
        for _, row in self.df.iterrows():
            node = row["node"]
            if "innovation" not in self.graph.nodes[node]:
                self.graph.nodes[node]["innovation"] = {}
            self.graph.nodes[node]["innovation"]["i_avg_discrete"] = {
                "citations": row["citations"],
                "slowdown": row["slowdown"]
            }

    def description(self):
        return (
            f"Innovation Loss (discrete slowdown): "
            f"Loss = N_R * I_avg * P = {self.nr} * {self.i_avg:.4f} * {self.p}\n"
            f"I_avg = media de slowdown aplicado por nodo según clase\n"
            f"Valores usados: {self.slowdown_values}"
        )
