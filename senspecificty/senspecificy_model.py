import networkx as nx

from relevance.database import Database


class SpecificitySensitivityModel:
    """
    Specificity and Sensitivity relevance model.
    """

    def __init__(self, graph: nx.DiGraph, database: Database):
        """
        Initialize the Specificity and Sensitivity relevance model.

        Parameters:
        graph (DiGraph): The directed graph.
        P_X1 (float): The probability of generating new knowledge.
        P_X0 (float): The probability of not generating new knowledge.
        P_S1_given_X1 (float): The probability of no lost information given new knowledge is generated.
        P_S0_given_X0 (float): The probability of lost information given no new knowledge is generated.
        """

        self.graph = graph
        self.database = database
        self.P_X1 = None
        self.P_X0 = None
        self.P_X0_S1 = None
        self.P_S1_X1 = None
        self.P_S0_X1 = None
        self.P_S1_X0 = None
        self.P_X1_S0 = None
        self.P_S0_X0 = None
        self.P_X1_S1 = None
        self.P_X0_S0 = None

    def estimate_probabilities(self):
        citing_edges = sum(1 for u, v in self.graph.edges)

        cited_papers_all_alive_references = sum(
            1 for node in self.graph.nodes
            if self.graph.in_degree(node) > 0 and self.database.is_paper_affected(node) == False
        )
        non_cited_papers_all_alive_references = sum(
            1 for node in self.graph.nodes
            if self.graph.in_degree(node) == 0 and self.database.is_paper_affected(node) == False
        )
        non_cited_papers_dead_references = sum(
            1 for node in self.graph.nodes
            if self.graph.in_degree(node) == 0 and self.database.is_paper_affected(node)
        )
        cited_papers_with_dead_references = sum(
            1 for node in self.graph.nodes
            if self.graph.in_degree(node) > 0 and self.database.is_paper_affected(node)
        )
        total_papers: int = self.database.count_total_articles()

        self.P_X1 = citing_edges / total_papers
        self.P_X0 = 1.0 - self.P_X1

        self.P_X1_S0 = cited_papers_with_dead_references / total_papers
        self.P_X0_S1 = non_cited_papers_all_alive_references / total_papers
        self.P_X1_S1 = cited_papers_all_alive_references / total_papers
        self.P_X0_S0 = non_cited_papers_dead_references / total_papers

    def get_probabilities(self):
        return self.P_X1, self.P_X0, self.P_X0_S1, self.P_X1_S1, self.P_X1_S0, self.P_X0_S0