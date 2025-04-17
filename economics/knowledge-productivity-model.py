class KnowledgeProductivityModel(EconomicLossModel):
    def __init__(self, data, a, l):
        super().__init__(data)
        self.a = a  # Eficiencia productiva
        self.l = l  # Número de investigadores

    def compute_loss(self):
        nnr = self.data['non_reproducible_papers']
        nt = self.data['total_papers']
        return self.a * self.l * (1 - nnr / nt)

    def description(self):
        return f"Productivity: ΔK = A * L * (1 - N_NR / N_T) = {self.a} * {self.l} * (1 - {nnr}/{nt})"
