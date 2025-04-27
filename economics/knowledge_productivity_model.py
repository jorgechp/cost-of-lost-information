from economics.economic_loss_model import EconomicLossModel


class KnowledgeProductivityModel(EconomicLossModel):
    def __init__(self, data, a, l):
        super().__init__(data)
        self.a = a  # Eficiencia productiva
        self.l = l  # Número de investigadores
        self.nnr = self.data['non_reproducible_papers']
        self.nt = self.data['total_papers']

    def compute_loss(self):
        return self.a * self.l * (1 - self.nnr / self.nt)

    def description(self):
        return f"Productivity: ΔK = A * L * (1 - N_NR / N_T) = {self.a} * {self.l} * (1 - {self.nnr}/{self.nt})"
