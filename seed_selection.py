import time
import numpy as np
import networkx as nx
from  structures.PriorityQueue import PriorityQueue

def random(graph, k):
    """
    Choose k nodes or k % of nodes randomly
    :param graph: the network
    :param k: if k is integer -> k is the number of nodes to add in seed set
        else it stands for the ratio of nodes in the seed set
    :return: the seed set
    """
    start = time.time()

    if k == 0:
        print("Seed set can not be empty!")
        return

    seed_set = list(np.random.choice(graph.nodes, int(k), replace=False))

    print("Seed set creation time:{}".format(time.time() - start))
    return seed_set


def all_aggressive(graph):
    """
        :return all aggressive users
    """

    aggression_score_threshold = 0.305  # gives 5594 users which is approximately 8% of total users

    start = time.time()

    aggressive_users = [node for node, attr in graph.nodes(data=True) if attr['aggression_score'] >= aggression_score_threshold]

    print("Seed set size:{}, time:{}".format(len(aggressive_users), time.time() - start))
    return aggressive_users


def top_aggressive(graph, k):
    """
        :return top k aggressive users
    """
    start = time.time()

    aggressive_users = {node: attr['aggression_score'] for node, attr in graph.nodes(data=True)
            if 'aggression_type' in attr and attr['aggression_type'] == 'offensive'}
    if k >= len(aggressive_users):
        return all_aggressive(graph)


    sorted_users = sorted(aggressive_users, key=aggressive_users.get, reverse=True)
    seed_set = sorted_users[:k]

    print("Seed set creation time:{}".format(time.time() - start))
    return seed_set


def top_central_aggressive_betweenness(graph, k):
    """
        :return top k aggressive users with respect to betweenness centrality
    """
    start = time.time()

    aggressive_users = [node for node, attr in graph.nodes(data=True)
                        if 'aggression_type' in attr and attr['aggression_type'] == 'offensive']
    if k >= len(aggressive_users):
        print("k should be smaller than the number of aggressive users ({})".format(len(aggressive_users)))
        return

    betweenness = nx.betweenness_centrality(graph)
    top_users = sorted(betweenness, key=betweenness.get, reverse=True)
    result = list()
    for user in top_users:
        if len(result) == k:
            break
        if user in aggressive_users:
            result.append(user)

    print("Seed set creation time:{}".format(time.time() - start))
    return result


def top_central_aggressive_eigenvector(graph, k):
    """
        :return top k aggressive users with respect to eigenvector centrality
    """
    start = time.time()

    aggressive_users = [node for node, attr in graph.nodes(data=True)
                        if 'aggression_type' in attr and attr['aggression_type'] == 'offensive']
    if k >= len(aggressive_users):
        print("k should be smaller than the number of aggressive users ({})".format(len(aggressive_users)))
        return

    eigenvector = nx.eigenvector_centrality(graph)
    top_users = sorted(eigenvector, key=eigenvector.get, reverse=True)
    result = list()
    for user in top_users:
        if len(result) == k:
            break
        if user in aggressive_users:
            result.append(user)

    print("Seed set creation time:{}".format(time.time() - start))
    return result


def single_discount(graph, k):
    """
    Worse than degree_discount but applicable both in LT and IC. O(k*log(n)+m)
    """
    start = time.time()

    seeds = list()
    degrees = PriorityQueue()
    for u in graph:
        degree = graph.out_degree(u)
        degrees.add_task(u, -degree)

    for i in range(k):
        u, priority = degrees.pop_item()
        seeds.append(u)
        for v in graph.neighbors(u):
            if v not in seeds:
                priority, count, task = degrees.entry_finder[v]

                # find edge weight
                weight = graph[u][v]['weight']
                degrees.add_task(v, priority + weight)  # discount degree by edge weight

    print("Seed set creation time:{}".format(time.time() - start))
    return seeds


def degree_discount(graph, k):
    """
    The best but only applicable in IC
    """
    start = time.time()

    seeds = []
    dd = PriorityQueue()  # degree discount
    t = dict()  # number of adjacent vertices that are in S
    degrees = dict()

    # initialize degree discount
    for u in graph.nodes():
        degrees[u] = graph.out_degree(u)
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd.add_task(u, -degrees[u])  # add degree of each node
        t[u] = 0

    # add vertices to S greedily
    for i in range(k):
        u, priority = dd.pop_item()  # extract node with maximal degree discount
        seeds.append(u)
        for v in graph.neighbors(u):
            if v not in seeds:
                p = graph[u][v]['weight']
                t[v] += 1  # increase number of selected neighbors
                priority = degrees[v] - 2 * t[v] - (degrees[v] - t[v]) * t[v] * p  # discount of degree
                dd.add_task(v, -priority)

    print("Seed set creation time:{}".format(time.time() - start))
    return seeds


