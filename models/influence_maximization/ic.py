from copy import deepcopy
import time
import networkx as nx
import random
import utility
import metrics


def run_ic(graph, s, args, instance, exp_type='modeling'):
    """
    Run an IC process until there are no nodes left to activate
    :param exp_type: type of experiment. modeling or blocking
    :param instance: current repeat of the experiment
    :param graph: the network
    :param s: initial seed set
    :param args: command-line arguments
    """

    assert type(graph) == nx.DiGraph, 'Graph G should be an instance of networkx.DiGraph'
    assert type(s) == list, 'Seed set S should be an instance of list'

    print('IC Model')

    start = time.time()

    agg_scores = list()

    seeds = deepcopy(s)  # contains all the currently activated nodes
    activations = deepcopy(s)   # keeps track of the new activations at each step
    print("Activated nodes: ", (len(seeds)))

    if args.snapshot:
        metrics.take_snapshot(graph, args, instance, 0, exp_type)

    aggression = utility.get_network_aggression(graph)
    print('Aggression: ', aggression)
    agg_scores.append(aggression)

    iteration = 1
    while len(activations):  # while there are new activated nodes
        activators = dict()  # contains new activations and the list of activator nodes. gets cleared on every step.
        for u in activations:
            for v in graph.neighbors(u):
                if v not in seeds:
                    edge = graph.get_edge_data(u, v, default=None)
                    weight = edge['weight']

                    coin_toss = random.random()
                    if coin_toss <= weight:  # v will be added in seeds at the end of the current step
                        if v in activators.keys():
                            activators[v].append(u)  # add u to v's activators
                        else:
                            activators.update({v: [u]})

        new_activations = list(activators.keys())
        seeds.extend(new_activations)  # add new activations to seeds
        activations = new_activations  # keep track of the new activations as long as activators dict will be cleared

        # transfer aggression
        if args.activation:
            utility.transfer_aggression(graph, args.activation, activators)

        print("Activated nodes: ", (len(seeds)))

        # Aggression score
        agg_score = utility.get_network_aggression(graph)
        print('Aggression: ', agg_score)
        agg_scores.append(agg_score)

        if args.snapshot:
            metrics.take_snapshot(graph, args, instance, iteration, exp_type)

        iteration += 1

    print()
    print('IC time:{}'.format(time.time() - start))
    print('Total activated nodes: ', (len(seeds)))
    print('Total aggression:', agg_scores[-1])

    return len(seeds), agg_scores
