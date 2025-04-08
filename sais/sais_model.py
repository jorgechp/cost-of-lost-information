import random

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')  # or 'Agg', 'Qt5Agg', etc.

class SAISModel:
    def __init__(self, graph, beta, gamma, delta):
        self.graph = graph
        self.beta = beta  # Infection rate
        self.gamma = gamma  # Recovery rate
        self.delta = delta  # Alert rate
        self.states = {node: 'S' for node in graph.nodes}

    def step(self):
        new_states = self.states.copy()
        for node in self.graph.nodes:
            if self.states[node] == 'S':
                neighbors = list(self.graph.successors(node))
                infected_neighbors = [n for n in neighbors if self.states[n] == 'I']
                if infected_neighbors and random.random() < self.beta:
                    new_states[node] = 'I'
                elif random.random() < self.delta:
                    new_states[node] = 'A'
            elif self.states[node] == 'A':
                if random.random() < self.beta:
                    new_states[node] = 'I'
            elif self.states[node] == 'I':
                if random.random() < self.gamma:
                    new_states[node] = 'S'
        self.states = new_states

    def simulate_epidemic(self, steps, infected_nodes):
        """Simula la propagación de la epidemia en la red."""
        # Infect initial nodes
        for node in infected_nodes:
            self.states[node] = 'A'  # Comienzan en estado Asintomático

        history = []
        history_states = []

        for _ in range(steps):
            state_counts = {state: sum(1 for node in self.graph.nodes if self.states[node] == state) for state in ['S', 'A', 'I']}
            history.append(state_counts)
            history_states.append(self.states.copy())
            self.step()

        return history, history_states

    def plot_epidemic(self, history):
        """Grafica la evolución de la epidemia."""
        time = range(len(history))
        s_vals = [h['S'] for h in history]
        a_vals = [h['A'] for h in history]
        i_vals = [h['I'] for h in history]

        plt.plot(time, s_vals, label='Health', color='blue')
        plt.plot(time, a_vals, label='Alert', color='orange')
        plt.plot(time, i_vals, label='Infected', color='red')
        plt.xlabel('Time')
        plt.ylabel('Number of nodes')
        plt.title(f'Epidemic Evolution (beta={self.beta}, gamma={self.gamma}, delta={self.delta})')
        plt.legend()
        plt.show()
