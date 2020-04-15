import networkx as nx
import os
from graph_creation import weight_schemes_utils as wsu


def create_graph_from_dataset(weights, power_type='frac'):
    """
    Creates a directed graph from the given dataset using a weighting method provided by the user
    :param weights: the type of edge weights ['random', 'constant', 'in_degree', 'jaccard', 'power', 'weighted_overlap']
    :param power_type: the type of power scores when the corresponding weighting strategies are selected
    :param p: value when constant weighting is selected as the scheme
    """

    graph = nx.read_edgelist("../../network/edgelist", create_using=nx.DiGraph, nodetype=int)

    if weights == 'jaccard':
        wsu.jaccard_overlap(graph)
    elif weights == 'power':
        wsu.power_score(graph, power_type)
    elif weights == 'weighted_overlap':
        wsu.weighted_overlap(graph, power_type)
    elif weights == 'random':  # edge weight = random in range [0,1]
        wsu.random(graph)
    else:
        return

    directory = '../data/graphs/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    nx.write_gpickle(graph, '../data/graphs/Twitter_{}'.format(weights))
