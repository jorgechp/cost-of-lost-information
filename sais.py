import argparse
import networkx as nx
from scipy.optimize import minimize

from models.abstract_tr import TrModel
from relevance.database import Database
from sais.sais_model import SAISModel
from transmission.affectation_transmission_model import AffectationTransmissionModel


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
    sais_model.plot_epidemic(history)
    fill_nodes_information(graph, history_states, beta, gamma, delta)

    print("Simulating epidemic with optimized parameters...")
    sais_model = SAISModel(graph, beta=beta_opt, gamma=gamma_opt, delta=delta_opt)
    history_opt, history_states_opt = sais_model.simulate_epidemic(args.steps, affected_nodes)
    print(history)
    sais_model.plot_epidemic(history)
    fill_nodes_information(graph, history_states_opt, beta_opt, gamma_opt, delta_opt)

    tr_model.save_graph(args.pickle_path)


if __name__ == "__main__":
    main()
