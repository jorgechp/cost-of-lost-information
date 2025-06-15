"""
SAIS (Susceptible-Alert-Infected-Susceptible) Model Simulator
==========================================================

This module implements the SAIS epidemic model for analyzing information spread
in citation networks. It includes optimization, visualization, and analysis tools.

Features:
---------
- SAIS model simulation with customizable parameters
- Parameter optimization using error minimization
- Multiple visualization methods:
  * Time evolution curves
  * Scenario comparisons
  * Parameter sensitivity heatmaps
- Node state tracking and analysis

Required Dependencies:
--------------------
- matplotlib
- seaborn
- numpy 
- scipy
- networkx (through TrModel)

Usage:
------
Command line:
    python sais.py path/to/graph.pkl -s number_of_steps
"""

import argparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from typing import Dict, List, Any

from transmission.transmission_model import TransmissionModel
from relevance.database import Database
from sais.sais_model import SAISModel


def error_function(params: List[float], tr_model: TransmissionModel,
                   affected_nodes: List[str], affected_levels: Dict[int, int],
                   steps: int) -> float:
    """
    Calculate error between predicted and actual affected nodes.

    Args:
        params: List containing [beta, gamma, delta] parameters
        tr_model: Transmission model instance
        affected_nodes: List of initially affected nodes
        affected_levels: Dictionary mapping levels to number of affected nodes
        steps: Number of simulation steps

    Returns:
        float: Sum of squared errors between prediction and actual values
    """
    beta, gamma, delta = params
    sais_model = SAISModel(tr_model.get_graph(), beta=beta, gamma=gamma, delta=delta)
    history = sais_model.simulate_epidemic(steps, affected_nodes)[0]

    # Calculate predicted levels
    predicted_levels = {level: 0 for level in affected_levels.keys()}
    for t, state_counts in enumerate(history):
        for level in affected_levels.keys():
            if t == level:
                predicted_levels[level] = state_counts['I'] + state_counts['A']

    # Calculate error
    error = sum((predicted_levels[level] - affected_levels[level]) ** 2 
                for level in affected_levels.keys())
    return error


def fill_nodes_information(graph: Any, history: List[Dict], 
                         beta: float, gamma: float, delta: float) -> None:
    """
    Store SAIS model simulation history in graph nodes.

    Args:
        graph: NetworkX graph object
        history: List of node states over time
        beta: Infection rate
        gamma: Recovery rate
        delta: Alert rate
    """
    for node in graph.nodes:
        if 'sais' not in graph.nodes[node]:
            graph.nodes[node]['sais'] = {}
        parameter_id = f"{beta}_{gamma}_{delta}"
        graph.nodes[node]['sais'][parameter_id] = {
            step: history[step][node] for step in range(len(history))
        }


def plot_sais_timecurves(history: List[Dict], filename: str = "figure_sais_timecurves.svg") -> None:
    """
    Plot time evolution of S, A, and I states.

    Args:
        history: List of dictionaries containing state counts
        filename: Output file path
    """
    steps = len(history)
    S = [state['S'] for state in history]
    A = [state['A'] for state in history]
    I = [state['I'] for state in history]

    plt.figure(figsize=(10, 5))
    plt.plot(range(steps), S, label="Susceptible (S)", color="blue")
    plt.plot(range(steps), A, label="Alert (A)", color="orange")
    plt.plot(range(steps), I, label="Infected (I)", color="red")

    plt.xlabel("Time Steps")
    plt.ylabel("Number of Nodes")
    plt.title("SAIS Model State Evolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def compare_sais_scenarios(history_opt: List[Dict], history_no_recovery: List[Dict], 
                         filename: str = "figure_sais_comparison.svg") -> None:
    """
    Compare different SAIS model scenarios.

    Args:
        history_opt: History with optimized parameters
        history_no_recovery: History without recovery
        filename: Output file path
    """
    I_opt = [h['I'] + h['A'] for h in history_opt]
    I_no_rec = [h['I'] + h['A'] for h in history_no_recovery]
    steps = range(len(I_opt))

    plt.figure(figsize=(10, 5))
    plt.plot(steps, I_opt, label="Optimized parameters", color="green")
    plt.plot(steps, I_no_rec, label="No recovery (γ=0)", color="purple", linestyle="--")
    plt.xlabel("Time Steps")
    plt.ylabel("Alert + Infected Nodes")
    plt.title("SAIS Model Scenario Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def generate_heatmap_sais(graph: Any, infected_nodes: List[str], 
                         gamma_fixed: float, steps: int,
                         filename: str = "figure_sais_heatmap.svg") -> None:
    """
    Generate heatmap for parameter sensitivity analysis.

    Args:
        graph: NetworkX graph object
        infected_nodes: Initially infected nodes
        gamma_fixed: Fixed recovery rate
        steps: Number of simulation steps
        filename: Output file path
    """
    beta_vals = np.linspace(0, 0.5, 10)
    delta_vals = np.linspace(0, 0.5, 10)
    heatmap_data = np.zeros((len(beta_vals), len(delta_vals)))

    for i, beta in enumerate(beta_vals):
        for j, delta in enumerate(delta_vals):
            model = SAISModel(graph, beta=beta, gamma=gamma_fixed, delta=delta)
            history, _ = model.simulate_epidemic(steps, infected_nodes)
            final_state = history[-1]
            infected_total = final_state['I'] + final_state['A']
            heatmap_data[i, j] = infected_total

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heatmap_data,
        xticklabels=[f"{d:.2f}" for d in delta_vals],
        yticklabels=[f"{b:.2f}" for b in beta_vals],
        cmap="YlOrRd",
        cbar_kws={'label': 'Final A + I Nodes'}
    )
    plt.xlabel("δ (Alert Rate)")
    plt.ylabel("β (Infection Rate)")
    plt.title(f"Infected + Alert Nodes after {steps} steps (γ fixed = {gamma_fixed})")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    """
    Main execution function for SAIS model analysis.
    """
    parser = argparse.ArgumentParser(
        description="SAIS model simulation for citation networks."
    )
    parser.add_argument(
        'pickle_path', 
        type=str, 
        help="Path to the pickle file containing the graph"
    )
    parser.add_argument(
        '-s', 
        type=int, 
        dest='steps', 
        help="Number of simulation steps"
    )

    args = parser.parse_args()

    # Initialize database and model
    db = Database()
    tr_model = TransmissionModel(db)
    tr_model.load_graph(args.pickle_path)

    # Get affected nodes and their distribution
    affected_nodes = tr_model.get_affected_nodes()
    affected_levels = tr_model.count_affected_levels(affected_nodes)
    print("\nAffected nodes per level:")
    for level, count in affected_levels.items():
        print(f"Level {level}: {count}")

    # Optimize parameters
    initial_params = [0.1, 0.1, 0.1]
    result = minimize(
        error_function, 
        initial_params, 
        args=(tr_model, affected_nodes, affected_levels, args.steps)
    )
    beta_opt, gamma_opt, delta_opt = result.x
    print(f"\nOptimized parameters:")
    print(f"β (infection rate) = {beta_opt:.4f}")
    print(f"γ (recovery rate)  = {gamma_opt:.4f}")
    print(f"δ (alert rate)     = {delta_opt:.4f}")

    # Run non-optimized simulation
    print("\nSimulating with baseline parameters (β=0.1, γ=0, δ=0.05)...")
    graph = tr_model.get_graph()
    sais_model = SAISModel(graph, beta=0.1, gamma=0, delta=0.05)
    history, history_states = sais_model.simulate_epidemic(args.steps, affected_nodes)
    fill_nodes_information(graph, history_states, 0.1, 0, 0.05)

    # Run optimized simulation
    print("Simulating with optimized parameters...")
    sais_model = SAISModel(graph, beta=beta_opt, gamma=gamma_opt, delta=delta_opt)
    history_opt, history_states_opt = sais_model.simulate_epidemic(args.steps, affected_nodes)
    fill_nodes_information(graph, history_states_opt, beta_opt, gamma_opt, delta_opt)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_sais_timecurves(history_opt)
    compare_sais_scenarios(history_opt, history)
    generate_heatmap_sais(graph, affected_nodes, gamma_opt, args.steps)

    # Save results
    tr_model.save_graph(args.pickle_path)
    print("\nAnalysis complete. Results saved.")


if __name__ == "__main__":
    main()