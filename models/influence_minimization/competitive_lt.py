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
        assert threshold == 'aggression' or threshold == 'power', "Threshold options: ['aggression', 'power', 'random']"

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


def run_lt(graph, ns, ps, args, instance):
    """
    Run an LT process until there are no nodes left to activate
    :param instance: current repeat of the experiment
    :param graph: the network
    :param ns: initial negative seed set
    :param ps: initial positive seed set
    :param args: command-line arguments
    """
    assert type(graph) == nx.DiGraph, 'Graph G should be an instance of networkx.DiGraph'
    assert type(ns) == list, 'Negative seed set nS should be an instance of list'
    assert type(ps) == list, 'Positive seed set pS should be an instance of list'

    print('competitive LT Model')

    start = time.time()

    thresholds = calculate_thresholds(graph, 'aggression')  # according to the best result of modeling phase

    agg_scores = list()

    n_seeds = deepcopy(ns)
    p_seeds = deepcopy(ps)

    n_activations = deepcopy(ns)
    p_activations = deepcopy(ps)

    n_activators = dict()
    p_activators = dict()

    n_influences = dict(zip(graph.nodes, [0] * len(graph.nodes)))
    p_influences = dict(zip(graph.nodes, [0] * len(graph.nodes)))

    print("Negative activated nodes before healing seeds: ", (len(n_seeds)))
    aggression = utility.get_network_aggression(graph)
    print('Aggression before healing seeds: ', aggression)
    print()
    agg_scores.append(aggression)

    if args.snapshot:
        metrics.take_snapshot(graph, args, instance, 0, 'minimization')

    healed = utility.heal_seeds(graph, n_seeds)

    # remove healed nodes from negative seed set
    n_seeds = [seed for seed in n_seeds if seed not in healed]
    n_activations = [seed for seed in n_activations if seed not in healed]

    print("Negative activated nodes after healing seeds: ", (len(n_seeds)))
    aggression = utility.get_network_aggression(graph)
    print('Aggression after healing seeds: ', aggression)
    print()
    agg_scores.append(aggression)

    hops = 0  # needed in case of decaying aggression transfer
    iteration = 1
    while len(n_activations) or len(p_activations):  # run as long as one cascade continues

        if len(n_activations):
            n_activators, n_influences, new_activations = lt_step(graph, n_activations, n_seeds, n_activators,
                                                                  n_influences, thresholds, healed)  # the step

            n_seeds.extend(new_activations)  # add new activations to negative seeds
            n_activations = deepcopy(new_activations)

            # transfer aggression. for v that is the sum of the aggression score of the neighbors that activated v, normalized
            for v in new_activations:
                activators_v = n_activators[v]
                sum_agg = 0
                for u in activators_v:
                    sum_agg += graph.nodes[u]['aggression_score']

                graph.nodes[v]['aggression_score'] = sum_agg / len(activators_v)

        if len(p_activations):
            hops += 1
            p_activators, p_influences, new_activations = lt_step(graph, p_activations, p_seeds, p_activators,
                                                                  p_influences, thresholds)  # the step

            p_seeds.extend(new_activations)  # add new activations to positive seeds
            p_activations = deepcopy(new_activations)

            # try to reduce aggression
            if args.healing:
                new_healed = utility.heal_aggression(graph, p_activators, args.healing, hops)

                # remove healed nodes from negative seed set
                n_seeds = [seed for seed in n_seeds if seed not in new_healed]
                n_activations = [seed for seed in n_activations if seed not in new_healed]
                healed.extend(new_healed)

        print("Negative activated nodes: ", (len(n_seeds)))

        # Aggression score
        agg_score = utility.get_network_aggression(graph)
        print('Aggression: ', agg_score)
        agg_scores.append(agg_score)

        if args.snapshot:
            metrics.take_snapshot(graph, args, instance, iteration, 'minimization')

        iteration += 1

    print()
    print('Competitive LT time:{}'.format(time.time() - start))
    print('Total negative activated nodes: ', (len(n_seeds)))
    print('Total positive activated nodes: ', (len(p_seeds)))
    print('Total aggression:', agg_scores[-1])

    return len(n_seeds), len(p_seeds), agg_scores


def lt_step(graph, activations, seeds, activators, influences, thresholds, healed=None):
    """
        A single lt step
        :param healed: if applied (negative cascade) try to activate nodes not included in this set.
        :param activators: structure containing neighbors that are currently affecting a node
        :param thresholds: thresholds for every node
        :param influences: structure that tracks weights of neighbors on nodes
        :param graph: the network
        :param activations: the new activations to be tested
        :param seeds: the whole seed set
        :return: dict of (activated node: list of activators) items
        """

    new_activations = set()
    for u in activations:
        for v in graph.neighbors(u):
            if healed:
                bool_activation = True if (v not in seeds) and (v not in healed) else False
            else:
                bool_activation = True if v not in seeds else False

            if bool_activation:
                edge = graph.get_edge_data(u, v, default=None)
                weight = edge['weight']

                # doesnt mean that v will get activated at this time step
                if v not in activators.keys():
                    activators.update({v: [u]})
                else:
                    activators[v].append(u)

                influences[v] += weight
                if influences[v] >= thresholds[v]:
                    new_activations.add(v)

    return activators, influences, new_activations