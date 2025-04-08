import argparse
import networkx as nx
from scipy.optimize import minimize

from models.abstract_tr import TrModel
from relevance.database import Database
from sais.sais_model import SAISModel
from senspecificty.senspecificy_model import SpecificitySensitivityModel
from transmission.affectation_transmission_model import AffectationTransmissionModel


def fill_nodes_information(graph, history, beta, gamma, delta):
    for node in graph.nodes:
        graph.nodes[node]['sais'] = {}
        parameter_id = "{}_{}_{}".format(beta, gamma, delta)
        graph.nodes[node]['sais'][parameter_id] = {
            step: history[step][node] for step in range(len(history))
        }

def main():
    parser = argparse.ArgumentParser(description="Transmission models for citation network.")
    parser.add_argument('pickle_path', type=str, help="The path to the pickle file containing the graph.")

    args = parser.parse_args()

    # Initialize the database
    db = Database()
    tr_model = TrModel(db)
    tr_model.load_graph(args.pickle_path)

    spmodel = SpecificitySensitivityModel(tr_model.get_graph(), db)
    spmodel.estimate_probabilities()
    P_X1, P_X0, P_X0_S1, P_X1_S1, P_X1_S0, P_X0_S0 = spmodel.get_probabilities()

    print(f"P_X1: {P_X1} - Probability of generating new knowledge")
    print(f"P_X0: {P_X0} - Probability of not generating new knowledge")
    print(f"P_X0_S1: {P_X0_S1} - Probability of not generating new knowledge given no lost information")
    print(f"P_X1_S1: {P_X1_S1} - Probability of generating new knowledge given no lost information")
    print(f"P_X1_S0: {P_X1_S0} - Probability of generating new knowledge given lost information")
    print(f"P_X0_S0: {P_X0_S0} - Probability of not generating new knowledge given lost information")

if __name__ == "__main__":
    main()
