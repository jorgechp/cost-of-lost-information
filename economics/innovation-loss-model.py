class InnovationLossModel(EconomicLossModel):
    def __init__(self, data, i_avg, p):
        super().__init__(data)
        self.i_avg = i_avg  # Innovaciones promedio por dataset
        self.p = p          # Probabilidad de innovación

    def compute_loss(self):
        nr = self.data['non_reproducible_refs']
        return nr * self.i_avg * self.p

    def description(self):
        return f"Innovation Loss = N_R * I_avg * P = {self.data['non_reproducible_refs']} * {self.i_avg} * {self.p}"
