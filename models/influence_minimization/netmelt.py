import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import itertools


def net_melt(graph, k):
    """
    Find the k edges to remove from the graph according to NetMelt algorithm
    :param graph: the network
    :param k: the number of edges to remove
    :return: the edges to remove
    """
    nodes = list(graph)
    edges = list(graph.edges())
    # Keep track of node indices
    node_to_index = {node: idx for idx, node in enumerate(nodes)}

    adj = nx.adjacency_matrix(graph)  # Adjacency matrix from graph G
    adj = adj.astype(float)

    # max eigenvalue, right eigenvector
    right_eigval, right_eigvect = sp.sparse.linalg.eigs(adj, k=1,  which='LR')
    # max eigenvalue, left eigenvector
    left_eigval, left_eigvect = sp.sparse.linalg.eigs(adj, k=1, which='LM')

    # We now check for negative values of the max left eigenvector

    if np.min(left_eigvect) < 0:
        left_eigvect = - left_eigvect
    if np.min(right_eigvect) < 0:
        right_eigvect = - right_eigvect

    # calculate the score of every edge
    scores = {}
    for edge in edges:
        scores[(edge[0], edge[1])] = float(left_eigvect[node_to_index[edge[0]]].real *
                                           right_eigvect[node_to_index[edge[1]]].real)

    # sort values in descending order and return the top k
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[:k]
