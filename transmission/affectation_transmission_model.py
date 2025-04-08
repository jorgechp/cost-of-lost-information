# transmission/models.py

import networkx as nx

from relevance.database import Database

EPSILON = 1e-6

class AffectationTransmissionModel:
    def __init__(self, citation_graph: nx.DiGraph, references_db: Database):
        self.graph: nx.DiGraph = citation_graph
        self.db: Database = references_db
        self.visited = {}

    def compute_affectation_transmitted(self, article, section_weights, threshold, by_section=True) -> float:
        affectation = 0.0
        references = self.db.get_paper_affectation(article)
        flat_internal_references = set()
        flat_external_references = {}

        if article in self.visited:
            return self.visited[article]

        if article not in self.graph.nodes:
            return 0

        if by_section:
            w_per_section = 1/(len(self.graph.nodes[article]['sections']) + EPSILON)
            for section in self.graph.nodes[article]['sections']:
                section_external_references = {}

                if section['section_name'] in references:
                    section_external_references = references[section['section_name']]
                affectation += w_per_section * self.compute_affectation_status(section['cited_references'], section_external_references, section_weights, threshold, by_section)
        else:

            for internal_references in self.graph.nodes[article]['sections']:
                flat_internal_references.update(internal_references['cited_references'])
            if article in flat_internal_references:
                flat_internal_references.remove(article)

            for section, refs in references.items():
                for ref, value in refs.items():
                    flat_external_references[ref] = value
            try:
                affectation = self.compute_affectation_status(flat_internal_references, flat_external_references, section_weights, threshold, by_section)
            except RecursionError as e:
                print(f"Recursion error at article {article}")
                affectation = 0

        self.visited[article] = affectation
        return affectation

    def remove_back_edges(self):
        """
        Detect cycles in the graph and remove back edges to eliminate cycles.

        Returns:
        nx.DiGraph: The graph with back edges removed.
        """
        cycles = list(nx.simple_cycles(self.graph))
        while len(cycles) > 0:
            for cycle in cycles:
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    if self.graph.has_edge(v, u):
                        self.graph.remove_edge(v, u)
                        break
            cycles = list(nx.simple_cycles(self.graph))

    def LTM(self, references, section_weights, threshold, by_section):
        total_references = len(references)
        references_weight = 1/(total_references + EPSILON)
        value = 0
        for reference in references:
            value += references_weight * self.compute_affectation_transmitted(reference, section_weights, threshold, by_section)
        return value >= threshold

    def compute_affectation_status(self, internal_references, external_references, section_weights, threshold, by_section):
        is_infected = False
        is_affected = False
        for reference, is_alive in external_references.items():
            is_infected = not is_alive

        if not is_infected and len(internal_references) > 0:
            is_affected = self.LTM(internal_references, section_weights, threshold, by_section)

        return 1 if is_infected or is_affected else 0

    def get_affectation_transmission_dict(self):
        return self.visited

# Example usage

    #
    # # Define section weights and threshold
    # section_weights = {
    #     "Introduction": 0.15,
    #     "Methodology": 0.30,
    #     "Results": 0.30,
    #     "Discussions/Conclusions": 0.25
    # }
    # threshold = 0.5
