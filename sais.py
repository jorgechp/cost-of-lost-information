import argparse
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from models.abstract_tr import TrModel
from relevance.database import Database
from sais.sais_model import SAISModel


def error_function(params, tr_model, affected_nodes, affected_levels, steps):
    beta, gamma, delta = params
    sais_model = SAISModel(tr_model.get_graph(), beta=beta, gamma=gamma, delta=delta)
    history = sais_model.simulate_epidemic(steps, affected_nodes)[0]

    # Convert history to levels
    predicted_levels = {level: 0 for level in affected_levels.keys()}
    for t, state_counts in enumerate(history):
        for level in affected_levels.keys():
            if t == level:
                predicted_levels[level] = state_counts['I'] + state_counts['A']

    # Calculate the error as the sum of squared differences
    error = sum((predicted_levels[level] - affected_levels[level]) ** 2 for level in affected_levels.keys())
    return error

def fill_nodes_information(graph, history, beta, gamma, delta):
    for node in graph.nodes:
        if 'sais' not in graph.nodes[node]:
            graph.nodes[node]['sais'] = {}
        parameter_id = "{}_{}_{}".format(beta, gamma, delta)
        graph.nodes[node]['sais'][parameter_id] = {
            step: history[step][node] for step in range(len(history))
        }


def plot_sais_timecurves(history, filename="figure_sais_timecurves.svg"):
    steps = len(history)
    S = [state['S'] for state in history]
    A = [state['A'] for state in history]
    I = [state['I'] for state in history]

    plt.figure(figsize=(10, 5))
    plt.plot(range(steps), S, label="Susceptibles (S)", color="blue")
    plt.plot(range(steps), A, label="Alertas (A)", color="orange")
    plt.plot(range(steps), I, label="Infectados (I)", color="red")

    plt.xlabel("Pasos temporales")
    plt.ylabel("Número de nodos")
    plt.title("Evolución de estados en el modelo SAIS")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def compare_sais_scenarios(history_opt, history_no_recovery, filename="figure_sais_comparison.svg"):
    I_opt = [h['I'] + h['A'] for h in history_opt]
    I_no_rec = [h['I'] + h['A'] for h in history_no_recovery]
    steps = range(len(I_opt))

    plt.figure(figsize=(10, 5))
    plt.plot(steps, I_opt, label="Optimized parameters", color="green")
    plt.plot(steps, I_no_rec, label="gamma=0", color="purple", linestyle="--")
    plt.xlabel("Steps")
    plt.ylabel("Nodes in state Alert or Infected")
    plt.title("Scenarios comparison in SAIS model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def generate_heatmap_sais(graph, infected_nodes, gamma_fixed, steps, filename="figure_sais_heatmap.svg"):
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
        cbar_kws={'label': 'Nodos A + I al final'}
    )
    plt.xlabel("δ (Tasa de alerta)")
    plt.ylabel("β (Tasa de infección)")
    plt.title(f"Infectados + Alertas tras {steps} pasos (γ fijo = {gamma_fixed})")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Transmission models for citation network.")
    parser.add_argument('pickle_path', type=str, help="The path to the pickle file containing the graph.")
    parser.add_argument('-s', type=int, dest='steps', help="Number of steps")

    args = parser.parse_args()

    # Initialize the database
    db = Database()
    tr_model = TrModel(db)
    tr_model.load_graph(args.pickle_path)


    # Identify affected nodes
    affected_nodes = tr_model.get_affected_nodes()

    affected_levels = tr_model.count_affected_levels(affected_nodes)
    print("Number of affected nodes at each level:")

    for level, node_count in affected_levels.items():
        print("Level {}: {}".format(level, node_count))

    initial_params = [0.1, 0.1, 0.1]
    result = minimize(error_function, initial_params, args=(tr_model, affected_nodes, affected_levels, args.steps))
    beta_opt, gamma_opt, delta_opt = result.x
    print(f"Optimized parameters: beta={beta_opt}, gamma={gamma_opt}, delta={delta_opt}")

    print("Simulating epidemic with non optimized parameters... beta=0.1, gamma=0, delta=0.05")
    graph = tr_model.get_graph()
    beta = 0.1
    gamma = 0
    delta = 0.05
    sais_model = SAISModel(graph, beta=beta, gamma=gamma, delta=delta)
    history, history_states = sais_model.simulate_epidemic(args.steps, affected_nodes)
    print(history)
    # sais_model.plot_epidemic(history)
    fill_nodes_information(graph, history_states, beta, gamma, delta)

    print("Simulating epidemic with optimized parameters...")
    sais_model = SAISModel(graph, beta=beta_opt, gamma=gamma_opt, delta=delta_opt)
    history_opt, history_states_opt = sais_model.simulate_epidemic(args.steps, affected_nodes)
    print(history)
    # sais_model.plot_epidemic(history)
    fill_nodes_information(graph, history_states_opt, beta_opt, gamma_opt, delta_opt)

    # Generar curva temporal del escenario optimizado
    print("Generando figura: evolución temporal con parámetros optimizados...")
    plot_sais_timecurves(history_opt, "figure_sais_timecurves.svg")

    # Generar comparación de escenarios
    print("Generando figura: comparación de escenarios...")
    compare_sais_scenarios(history_opt, history, "figure_sais_comparison.svg")

    print("Generando mapa de calor para análisis de sensibilidad...")
    generate_heatmap_sais(tr_model.get_graph(), affected_nodes, gamma_fixed=gamma_opt, steps=args.steps, filename="figure_sais_heatmap.svg")


    tr_model.save_graph(args.pickle_path)


if __name__ == "__main__":
    main()
