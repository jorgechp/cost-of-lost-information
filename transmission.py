import argparse
import networkx as nx

from models.abstract_tr import TrModel
from relevance.database import Database
from transmission.affectation_transmission_model import AffectationTransmissionModel


def main():
    parser = argparse.ArgumentParser(description="Transmission models for citation network.")
    parser.add_argument('pickle_path', type=str, help="The path to the pickle file containing the graph.")
    parser.add_argument('-s', action='store_true', dest='by_section', help="Analyze by sections.")
    parser.add_argument('-t', dest='thresholds', type=float, nargs='+',
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help="List of thresholds to process.")

    args = parser.parse_args()



    # Initialize the database
    db = Database()
    tr_model = TrModel(db)
    graph = tr_model.load_graph(args.pickle_path)

    # Identify affected nodes
    affected_nodes = tr_model.get_affected_nodes()

    affected_levels = tr_model.count_affected_levels(affected_nodes)
    print("Number of affected nodes at each level:")

    for level, node_count in affected_levels.items():
        print("Level {}: {}".format(level, node_count))

    section_weights = {
        "Introduction": 0.15,
        "Methodology": 0.30,
        "Results": 0.30,
        "Discussions/Conclusions": 0.25
    }

    for t in args.thresholds:
        model = AffectationTransmissionModel(graph, db)
        model.remove_back_edges()

        pending_nodes = []

        for node in graph.nodes:
            node_value = model.compute_affectation_transmitted(node, section_weights, t, by_section=args.by_section)
            if node_value > 0:
                pending_nodes.append(node)

        affectation_transmission = model.get_affectation_transmission_dict()
        print(f"Threshold: {t}")
        print(f"Affected articles: {len(affectation_transmission)}")

        level = 0

        while len(pending_nodes) > 0:
            target_nodes = pending_nodes
            print(f"Level {level}: {len(target_nodes)}")
            pending_nodes = []
            for node in target_nodes:
                descendants = list(nx.descendants(graph, node))
                affected_descendants = [desc for desc in descendants if affectation_transmission.get(desc, False)]
                pending_nodes.extend(affected_descendants)
            level += 1

        # iterates all nodes in graph
        for node in graph.nodes:
            graph.nodes[node]['transmission_value'] = {}
            graph.nodes[node]['transmission_value'][t] = affectation_transmission.get(node, 0)

    # save the graph with transmission values
    tr_model.save_graph(args.pickle_path)


if __name__ == "__main__":
    main()
