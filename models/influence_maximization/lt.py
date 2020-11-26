from copy import deepcopy
import time
import networkx as nx
from numpy import random
from graph_creation import weight_schemes_utils as wsu
import utility
import metrics


def calculate_thresholds(graph, threshold):
    """
    Compute node thresholds according to the given strategy
    """

    node_thresholds = dict()

    if type(threshold) == str:

        if threshold == 'aggression':
            print('Using aggression scores as thresholds')
            for node in graph.nodes:
                node_thresholds[node] = graph.nodes[node]['aggression_score']
        elif threshold == 'power':  # power scores
            print('Using power scores as thresholds')
            power_scores = wsu.node_power_scores(graph)
            for node in graph.nodes:
                node_thresholds[node] = power_scores[node]
        elif threshold == 'random': # random
            print('Node thresholds not given! Picking them uniformly from [0,1]...')
            for node in graph.nodes:
                node_thresholds[node] = random.uniform()
        else:
            return
    elif type(threshold) == float:
        print('Constant node threshold was given. Setting all thresholds to {} ...'.format(threshold))
        for node in graph.nodes:
            node_thresholds[node] = threshold

    elif type(threshold) == dict:
        assert len(threshold.keys()) == len(graph.nodes), 'All nodes should be included in the threshold dictionary'

        for node in graph.nodes:
            assert node in threshold.keys(), 'Only graph nodes should be included in the threshold dictionary'

        print('Setting node thresholds from dictionary...')
        node_thresholds = threshold

    return node_thresholds


def run_lt(graph, s, args, instance=None, exp_type='modeling'):
    """
    Run an LT process until there are no nodes left to activate
    :param exp_type: type of experiment. modeling or blocking
    :param instance: current repeat of the experiment
    :param graph: the network
    :param s: initial seed set
    :param args: command-line arguments
    """
    assert type(graph) == nx.DiGraph, 'Graph G should be an instance of networkx.DiGraph'
    assert type(s) == list, 'Seed set S should be an instance of list'

    print('LT Model')

    start = time.time()

    thresholds = calculate_thresholds(graph, args.threshold)

    agg_scores = list()

    seeds = deepcopy(s)
    activations = deepcopy(s)
    print("Activated nodes: ", (len(seeds)))

    activators = dict()
    node_influences = dict(zip(graph.nodes, [0] * len(graph.nodes)))

    if args.snapshot:
        metrics.take_snapshot(graph, args, instance, 0, exp_type)

    aggression = utility.get_network_aggression(graph)
    print('Aggression: ', aggression)
    agg_scores.append(aggression)

    iteration = 1
    while len(activations):  # while there are new activated nodes
        new_activations = set()
        for u in activations:
            for v in graph.neighbors(u):
                if v not in seeds:
                    edge = graph.get_edge_data(u, v, default=None)
                    weight = edge['weight']

                    # doesnt mean that v will get activated at this time step
                    if v not in activators.keys():
                        activators.update({v: [u]})
                    else:
                        activators[v].append(u)

                    node_influences[v] += weight
                    if node_influences[v] >= thresholds[v]:
                        new_activations.add(v)

        seeds.extend(new_activations)  # add new activations to seeds
        activations = deepcopy(new_activations)

        # transfer aggression. for v that is the sum of the aggression score of the neighbors that activated v, normalized
        for v in new_activations:
            activators_v = activators[v]
            sum_agg = 0
            for u in activators_v:
                sum_agg += graph.nodes[u]['aggression_score']

            graph.nodes[v]['aggression_score'] = sum_agg / len(activators_v)

        print("Activated nodes: ", (len(seeds)))

        # Aggression score
        agg_score = utility.get_network_aggression(graph)
        print('Aggression: ', agg_score)
        agg_scores.append(agg_score)

        if args.snapshot:
            metrics.take_snapshot(graph, args, instance, iteration, exp_type)
        iteration += 1

    print()
    print("LT time:{}".format(time.time() - start))
    print('Total activated nodes: ', (len(seeds)))
    print('Total aggression:', agg_scores[-1])

    return len(seeds), agg_scores
