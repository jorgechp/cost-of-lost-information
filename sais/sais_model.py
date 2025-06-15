"""
SAIS Epidemic Model
==================

This module implements the Susceptible-Alert-Infected-Susceptible (SAIS) model
for simulating epidemic spread in networks.

Features:
---------
- State transitions (S → A → I → S)
- Configurable transition rates
- Network-based spread simulation
- Temporal evolution tracking
- Visualization capabilities

States:
-------
S: Susceptible (healthy, can be infected)
A: Alert (aware of disease, reduced infection probability)
I: Infected (can spread to others)

Required Dependencies:
--------------------
- networkx
- matplotlib
- random
"""

import random
from typing import Dict, List, Set, Tuple, Any
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class ModelParameters:
    """Container for model parameters."""
    beta: float   # Infection rate
    gamma: float  # Recovery rate
    delta: float  # Alert rate


@dataclass
class StateCount:
    """Container for state counts."""
    susceptible: int
    alert: int
    infected: int

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary format."""
        return {
            'S': self.susceptible,
            'A': self.alert,
            'I': self.infected
        }


class SAISModel:
    """
    Implementation of the SAIS epidemic model.

    This model simulates disease spread with three states:
    - Susceptible (S)
    - Alert (A)
    - Infected (I)
    """

    def __init__(self,
                 graph: nx.Graph,
                 beta: float,
                 gamma: float,
                 delta: float):
        """
        Initialize SAIS model.

        Args:
            graph: Network structure
            beta: Infection rate (0-1)
            gamma: Recovery rate (0-1)
            delta: Alert rate (0-1)
        """
        if not (0 <= beta <= 1 and 0 <= gamma <= 1 and 0 <= delta <= 1):
            raise ValueError("Rates must be between 0 and 1")

        self.graph = graph
        self.params = ModelParameters(beta, gamma, delta)
        self.states: Dict[Any, str] = {
            node: 'S' for node in graph.nodes
        }

    def get_state_counts(self) -> StateCount:
        """
        Count nodes in each state.

        Returns:
            StateCount object with current counts
        """
        counts = {'S': 0, 'A': 0, 'I': 0}
        for state in self.states.values():
            counts[state] += 1

        return StateCount(
            counts['S'],
            counts['A'],
            counts['I']
        )

    def step(self) -> None:
        """Perform one time step of the simulation."""
        new_states = self.states.copy()

        for node in self.graph.nodes:
            current_state = self.states[node]

            if current_state == 'S':
                # Susceptible nodes can become infected or alert
                neighbors = list(self.graph.successors(node))
                infected_neighbors = [
                    n for n in neighbors
                    if self.states[n] == 'I'
                ]

                if infected_neighbors and random.random() < self.params.beta:
                    new_states[node] = 'I'
                elif random.random() < self.params.delta:
                    new_states[node] = 'A'

            elif current_state == 'A':
                # Alert nodes can become infected (with reduced probability)
                if random.random() < self.params.beta * 0.5:  # Reduced infection rate
                    new_states[node] = 'I'

            elif current_state == 'I':
                # Infected nodes can recover
                if random.random() < self.params.gamma:
                    new_states[node] = 'S'

        self.states = new_states

    def simulate(self,
                 steps: int,
                 initial_infected: Set[Any],
                 initial_alert: Set[Any] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Run epidemic simulation.

        Args:
            steps: Number of time steps
            initial_infected: Initially infected nodes
            initial_alert: Initially alert nodes (optional)

        Returns:
            Tuple of (state counts history, state maps history)
        """
        # Initialize states
        for node in initial_infected:
            self.states[node] = 'I'

        if initial_alert:
            for node in initial_alert:
                self.states[node] = 'A'

        history = []
        state_history = []

        # Run simulation
        for _ in range(steps):
            counts = self.get_state_counts()
            history.append(counts.to_dict())
            state_history.append(self.states.copy())
            self.step()

        return history, state_history

    def plot_evolution(self,
                       history: List[Dict],
                       title: str = None,
                       save_path: str = None) -> None:
        """
        Plot epidemic evolution over time.

        Args:
            history: Simulation history
            title: Plot title (optional)
            save_path: Path to save plot (optional)
        """
        time = range(len(history))
        s_counts = [h['S'] for h in history]
        a_counts = [h['A'] for h in history]
        i_counts = [h['I'] for h in history]

        plt.figure(figsize=(10, 6))
        plt.plot(time, s_counts,
                 label='Susceptible',
                 color='blue',
                 linestyle='-')
        plt.plot(time, a_counts,
                 label='Alert',
                 color='orange',
                 linestyle='--')
        plt.plot(time, i_counts,
                 label='Infected',
                 color='red',
                 linestyle='-.')

        plt.xlabel('Time Steps')
        plt.ylabel('Number of Nodes')
        plt.title(title or f'SAIS Model Evolution (β={self.params.beta}, γ={self.params.gamma}, δ={self.params.delta})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def get_metrics(self, history: List[Dict]) -> Dict[str, float]:
        """
        Calculate epidemic metrics.

        Args:
            history: Simulation history

        Returns:
            Dictionary of metrics
        """
        total_nodes = len(self.graph)
        max_infected = max(h['I'] for h in history)
        final_infected = history[-1]['I']
        outbreak_duration = len(history)

        return {
            'peak_infection_rate': max_infected / total_nodes,
            'final_infection_rate': final_infected / total_nodes,
            'outbreak_duration': outbreak_duration,
            'average_infected': sum(h['I'] for h in history) / len(history) / total_nodes
        }


def main():
    """Example usage of the SAIS model."""
    # Create sample network
    G = nx.erdos_renyi_graph(100, 0.1)

    # Initialize model
    model = SAISModel(
        graph=G,
        beta=0.3,   # Infection rate
        gamma=0.1,  # Recovery rate
        delta=0.05  # Alert rate
    )

    # Run simulation
    history, _ = model.simulate(
        steps=50,
        initial_infected={0, 1, 2}  # Start with nodes 0,1,2 infected
    )

    # Plot results
    model.plot_evolution(history)

    # Print metrics
    metrics = model.get_metrics(history)
    print("\nEpidemic Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()