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
