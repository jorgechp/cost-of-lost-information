from abc import ABC, abstractmethod


class EconomicLossModel(ABC):
    def __init__(self, data):
        """
        data: estructura con información básica para el modelo,
              como nodos afectados, número total de papers, etc.
        """
        self.data = data

    @abstractmethod
    def compute_loss(self):
        """Calcula la pérdida estimada de conocimiento"""
        pass

    @abstractmethod
    def description(self):
        """Devuelve una breve descripción del modelo y sus parámetros"""
        pass
