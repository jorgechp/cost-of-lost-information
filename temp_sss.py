import pickle
import networkx as nx

with open("output/openalex_citation_network_formatted.pickle", "rb") as f:
    G_unarxive = pickle.load(f)

with open("output/openalex_citation_network.pkl", "rb") as f:
    G_openalex = pickle.load(f)

print("UNARXIVE:")
print("  nodos:", G_unarxive.number_of_nodes())
print("  aristas:", G_unarxive.number_of_edges())
print("  ejemplo nodo:", list(G_unarxive.nodes(data=True))[0])

print("\nOPENALEX:")
print("  nodos:", G_openalex.number_of_nodes())
print("  aristas:", G_openalex.number_of_edges())
print("  ejemplo nodo:", list(G_openalex.nodes(data=True))[0])
