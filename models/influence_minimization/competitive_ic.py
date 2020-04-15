from copy import deepcopy
import time
import networkx as nx
import random
import utility
import metrics


def run_ic(graph, ns, ps, args, instance):
    """
    Run an IC process until there are no nodes left to activate
    :param instance: current repeat of the experiment
    :param graph: the network
    :param ns: initial negative seed set
    :param ps: initial positive seed set
    :param args: command-line arguments
    """

    assert type(graph) == nx.DiGraph, 'Graph G should be an instance of networkx.DiGraph'
    assert type(ns) == list, 'Negative seed set nS should be an instance of list'
    assert type(ps) == list, 'Positive seed set pS should be an instance of list'

    print('competitive IC Model')

    start = time.time()

    agg_scores = list()

    n_seeds = deepcopy(ns)
    p_seeds = deepcopy(ps)

    n_activations = deepcopy(ns)
    p_activations = deepcopy(ps)

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

        if len(n_activations):  # negative cascade single step
            n_activators = ic_step(graph, n_activations, n_seeds, healed)  # the step

            # add new activations to seed and track sets
            new_activations = list(n_activators.keys())
            n_seeds.extend(new_activations)
            n_activations = new_activations

            # transfer aggression of negative cascade according to best configuration of aggression modeling experiments
            # activation criterion was shown not to affect the efficiency of the model. Chose 'cumulative' randomly
            utility.transfer_aggression(graph, 'cumulative', n_activators)

        if len(p_activations):
            hops += 1
            p_activators = ic_step(graph, p_activations, p_seeds)

            new_activations = list(p_activators.keys())
            p_seeds.extend(new_activations)
            p_activations = new_activations

            # try to reduce aggression according to the positive cascade and the activation strategy
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
    print('Competitive IC time:{}'.format(time.time() - start))
    print('Total negative activated nodes: ', (len(n_seeds)))
    print('Total positive activated nodes: ', (len(p_seeds)))
    print('Total aggression:', agg_scores[-1])

    return len(n_seeds), len(p_seeds), agg_scores


def ic_step(graph, activations, seeds, healed=None):
    """
    A single ic step
    :param healed: if applied (negative cascade) try to activate nodes not included in this set.
    :param graph: the network
    :param activations: the new activations to be tested
    :param seeds: the whole seed set
    :return: dict of (activated node: list of activators) items
    """
    activators = dict()
    for u in activations:
        for v in graph.neighbors(u):
            if healed:
                bool_activation = True if (v not in seeds) and (v not in healed) else False
            else:
                bool_activation = True if v not in seeds else False

            if bool_activation:
                edge = graph.get_edge_data(u, v, default=None)
                weight = edge['weight']

                coin_toss = random.random()
                if coin_toss <= weight:  # v will be added in seeds at the end of the current step
                    if v in activators.keys():
                        activators[v].append(u)
                    else:
                        activators.update({v: [u]})

    return activators