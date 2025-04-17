class OpportunityCostModel(EconomicLossModel):
    def __init__(self, data, po, ik):
        super().__init__(data)
        self.po = po  # Probabilidad de generar nuevo conocimiento
        self.ik = ik  # Impacto medio del nuevo conocimiento

    def compute_loss(self):
        nr = self.data['non_reproducible_refs']
        return nr * self.po * self.ik

    def description(self):
        return f"Opportunity Cost: Loss = N_R * P_O * I_K = {self.data['non_reproducible_refs']} * {self.po} * {self.ik}"
