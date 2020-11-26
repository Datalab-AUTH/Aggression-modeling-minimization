import networkx as nx
import scipy as sp
from tqdm import tqdm


# k = number of influencers i want to find. no more than 100, too slow
def net_shield(graph, k):
    # G = network(filename)
    nodes = list(graph)
    n = len(nodes)
    adj = nx.adjacency_matrix(graph)  # Adjacency matrix from graph G
    adj = adj.astype(float)
    eigval, eigvect = sp.sparse.linalg.eigs(adj, k=1, which='LM')  # Calculate largest eigenvalue and corresponding eigenvector
    seeds = []
    v = []
    temp_s = []

    for j in range(n):
        v.append((2*eigval - adj[j,j])*eigvect[j]*eigvect[j])
    score = []
    for _ in tqdm(range(k)):
        del score[:]
        B = adj[:, temp_s]
        b = B*eigvect[temp_s]
        for j in range(n):
            if j in temp_s:
                score.append(-1)
            else:
                score.append(v[j]-2*b[j]*eigvect[j])
        m = max(score)
        maxs = [nodes[i] for i, j in enumerate(score) if j == m]
        temp = [i for i, j in enumerate(score) if j == m]
        seeds += maxs
        if len(seeds) > k:
            del seeds[k:]
        temp_s += temp
    return seeds

g = nx.read_gpickle('../../data/graphs/Twitter_jaccard')
import utility
g = nx.DiGraph(utility.insert_aggression(g, '../../../data/metrics.predictions.csv'))

# NORMAL
# nodes_to_remove = net_shield(g, 5594)
# in_edges = list(g.in_edges(nodes_to_remove))
# out_edges = list(g.out_edges(nodes_to_remove))


# AGGRESSION
graph = nx.DiGraph()
for edge in tqdm(g.edges(data=True)):
    u = edge[0]
    v = edge[1]
    agg_u = g.nodes[u]['aggression_score']
    agg_v = g.nodes[v]['aggression_score']
    graph.add_edge(u, v, weight=(agg_u * agg_v))
nodes_to_remove = net_shield(graph, 5594)
in_edges = list(graph.in_edges(nodes_to_remove))
out_edges = list(graph.out_edges(nodes_to_remove))


edges = list()
edges.extend(in_edges)
edges.extend(out_edges)

#create an undirected graph to remove duplicate edges, as the initial graph was directed
undirected = nx.Graph()
undirected.add_edges_from(edges)

print(len(in_edges), len(out_edges), len(edges), undirected.number_of_edges())

#5594 nodes -> 476690 edges for normal adjacency