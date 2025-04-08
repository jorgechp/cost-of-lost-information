import networkx as nx

class PageRankLossCalculator:
    def __init__(self, graph):
        self.graph = graph

    def compute_standard_pagerank(self):
        return nx.pagerank(self.graph, alpha=0.85)

    def compute_weighted_pagerank(self, missing_refs, penalized_weight=0.1, alpha=0.85):
        # Penalizar las aristas que apuntan a referencias perdidas
        G_weighted = self.graph.copy()
        for u, v in G_weighted.edges():
            if v in missing_refs:
                G_weighted[u][v]['weight'] = penalized_weight
            else:
                G_weighted[u][v]['weight'] = 1.0
        return nx.pagerank(G_weighted, alpha=alpha, weight='weight')

    def compute_personalized_pagerank(self, missing_refs, penalization_value=0.01):
        personalization = {
            node: penalization_value if node in missing_refs else 1.0
            for node in self.graph.nodes()
        }
        # Normalizar
        total = sum(personalization.values())
        personalization = {k: v / total for k, v in personalization.items()}
        return nx.pagerank(self.graph, alpha=0.85, personalization=personalization)

# Example usage
if __name__ == "__main__":
    # Create a sample directed graph
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (2, 4), (3, 1)])

    # List of affected nodes
    affected_nodes = [2, 3]

    # Initialize the PageRank loss calculator
    calculator = PageRankLossCalculator(G, affected_nodes)

    # Calculate and print the PageRank loss
    pagerank_loss = calculator.calculate_pagerank_loss()
    print("PageRank Loss:", pagerank_loss)