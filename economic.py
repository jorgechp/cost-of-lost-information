import argparse
import pickle

from economics.innovation_loss_model import InnovationLossModel
from economics.innovation_loss_model_discrete import InnovationLossModelDiscrete
from economics.knowledge_productivity_model import KnowledgeProductivityModel
from economics.opportunity_cost_model import OpportunityCostModel
from economics.pigou_loss_model import PigouKnowledgeLossModel
from economics.pigou_propagated_model import PigouPropagatedModel
from models.abstract_tr import TrModel
from relevance.database import Database

import arxiv

def get_authors_from_arxiv(arxiv_id):
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])

    try:
        results = client.results(search)
        result = next(results)
        return [author.name for author in result.authors]
    except Exception as e:
        return []

def estimate_po(graph, reproducible_nodes):
    cited_reproducibles = [
        node for node in reproducible_nodes if list(graph.successors(node))
    ]
    if not reproducible_nodes:
        return 0.0
    return len(cited_reproducibles) / len(reproducible_nodes)


def estimate_ik(graph, reproducible_nodes, max_depth=2):
    def count_descendants(node):
        visited = set()
        frontier = {node}
        for _ in range(max_depth):
            next_frontier = set()
            for n in frontier:
                children = set(graph.successors(n)) - visited
                next_frontier |= children
                visited |= children
            frontier = next_frontier
        return len(visited)

    if not reproducible_nodes:
        return 0.0
    ik_values = [count_descendants(n) for n in reproducible_nodes]
    return sum(ik_values) / len(ik_values)


def main():
    parser = argparse.ArgumentParser(description="Economic models for citation network.")
    parser.add_argument('pickle_path', type=str, help="The path to the pickle file containing the graph.")
    args = parser.parse_args()

    # Initialize database and load graph
    db = Database()
    tr_model = TrModel(db)
    tr_model.load_graph(args.pickle_path)
    graph = tr_model.graph

    pigou_model = PigouKnowledgeLossModel(data={
        "graph": graph,
        "db": db,
        "non_reproducible_refs": len(db.get_all_affected_papers())
    }, c_s=1.0)

    loss_pigou = pigou_model.compute_loss()
    print(pigou_model.description())
    print("Estimated Pigouvian loss:", loss_pigou)

    model = PigouPropagatedModel(data={
        "graph": graph,
        "db": db,
        "non_reproducible_refs": len(db.get_all_affected_papers())
    }, max_depth=3, decay="linear", citation_weight=True)

    loss = model.compute_loss()
    print(model.description())


    # Get affected and reproducible nodes
    affected = db.get_all_affected_papers()
    reproducible = set(graph.nodes())

    # Estimate economic parameters from the graph
    po = estimate_po(graph, reproducible)
    ik = estimate_ik(graph, reproducible, max_depth=2)
    nnr = len(affected)
    nt = len(graph.nodes())

    print(f"Estimated P_O: {po:.4f}")
    print(f"Estimated I_K: {ik:.4f}")
    print(f"Total papers: {nt}, Non-reproducible: {nnr}")

    unique_authors = set()
    for author_parts in data.get("authors", []):
        if isinstance(author_parts, list):
            # Filtra elementos vacíos y une con espacio
            author_name = " ".join(part for part in author_parts if part)
            unique_authors.add(author_name)
        else:
            unique_authors.add(str(author_parts))

    L = len(unique_authors)

    data = {
        'non_reproducible_refs': nnr,
        'non_reproducible_papers': nnr,  # as proxy
        'total_papers': nt
    }

    models = [
        OpportunityCostModel(data, po=po, ik=ik),
        KnowledgeProductivityModel(data, a=ik, l=L),  # A = ik as knowledge productivity proxy
    ]

    print("\n--- Economic Impact Models ---\n")
    for model in models:
        print(model.description())
        print("Estimated loss:", model.compute_loss())
        print()

    # Calcular modelos de innovación
    softmax_model = InnovationLossModel(data={
        "graph": graph,
        "db": db,
        "non_reproducible_refs": nnr
    }, p=po)

    discrete_model = InnovationLossModelDiscrete(data={
        "graph": graph,
        "db": db,
        "non_reproducible_refs": nnr
    }, p=po)

    softmax_loss = softmax_model.compute_loss()
    discrete_loss = discrete_model.compute_loss()

    print("\n--- Innovation Loss Models Comparison ---\n")

    print("[Softmax model]")
    print(softmax_model.description())
    print(f"Estimated loss: {softmax_loss:.2f}\n")

    print("[Discrete model]")
    print(discrete_model.description())
    print(f"Estimated loss: {discrete_loss:.2f}\n")

    diff = softmax_loss - discrete_loss
    print(f"Difference (softmax - discrete): {diff:.2f}")


if __name__ == "__main__":
    main()
