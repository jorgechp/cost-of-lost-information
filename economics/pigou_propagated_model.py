from .economic_loss_model import EconomicLossModel

class PigouPropagatedModel(EconomicLossModel):
    def __init__(self, data, max_depth=3, decay="linear", citation_weight=False):
        """
        Modelo Pigouviano con propagación de pérdida a través de la red.

        Parameters:
            data: debe contener 'graph', 'db', 'non_reproducible_refs'
            max_depth: profundidad máxima de propagación
            decay: "linear" (1/depth), "none", o función personalizada
            citation_weight: si True, pondera por nº de citas del nodo receptor
        """
        super().__init__(data)
        self.graph = data["graph"]
        self.db = data["db"]
        self.nr = data["non_reproducible_refs"]
        self.max_depth = max_depth
        self.decay = decay
        self.citation_weight = citation_weight
        self.loss = None

    def compute_loss(self):
        graph = self.graph
        db = self.db
        affected_nodes = db.get_all_affected_papers()
        loss = 0

        for node in affected_nodes:
            if node not in graph:
                continue
            visited = set()
            frontier = {node}

            for depth in range(1, self.max_depth + 1):
                next_frontier = set()
                for n in frontier:
                    children = set(graph.successors(n)) - visited
                    for child in children:
                        # Decay por profundidad
                        if self.decay == "linear":
                            depth_weight = 1 / depth
                        elif self.decay == "none":
                            depth_weight = 1
                        else:
                            depth_weight = self.decay(depth)  # si es una función

                        # Ponderación por número de citas
                        citation_factor = len(list(graph.successors(child))) if self.citation_weight else 1

                        loss += depth_weight * citation_factor
                    visited |= children
                    next_frontier |= children
                frontier = next_frontier

        self.loss = loss
        return self.loss

    def description(self):
        decay_desc = "1/depth" if self.decay == "linear" else "no decay"
        citation_desc = "weighted by citations" if self.citation_weight else "equal weight"
        loss_str = f"{self.loss:.2f}" if self.loss is not None else "0.00"
        return (
            f"Pigou-Propagated Loss Model:\n"
            f"- Depth: {self.max_depth}, Decay: {decay_desc}, Citations: {citation_desc}\n"
            f"- Estimated propagated loss from affected nodes: {loss_str}"
        )
