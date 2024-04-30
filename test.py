import networkx as nx
from bayanpy.BayanImplied import bayan

# H=nx.florentine_families_graph()
# approximate_output = bayan(H, develop_mode=True)

# G = nx.ring_of_cliques(4,5)
# approximate_output = bayan(G, develop_mode=True)


G = nx.powerlaw_cluster_graph(30, 20, 0.5, seed=16)
approximate_output = bayan(G, threshold=0.001, develop_mode=True)

