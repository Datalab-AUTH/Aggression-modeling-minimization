from numpy import random, power
import networkx as nx
import pandas as pd
import random as rn
import csv
import os.path


def heal_seeds(graph, seeds):
    """
    Healing strategy for seed nodes of positive cascade
    The higher their aggression score the harder to heal them
    Healing means agg score = 0
    :param graph: the network
    :param seeds: the positive seed set
    :return the nodes that were healed in order to remove them from negative seed set if they exist in there
    """

    healed = list()
    for v in seeds:
        coin_toss = rn.random()
        if coin_toss <= graph.nodes[v]['aggression_score']:  # v will be added in seeds at the end of the current step
            graph.nodes[v]['aggression_score'] = 0
            healed.append(v)

    return healed


def heal_aggression(graph, activators, healing, hops):
    """
    Apply aggression transfer strategy during a diffusion process
    :param hops: the number of hops currently done in the process. Needed in case of decaying aggression transfer
    :param graph: the network
    :param healing: how to transfer aggression to newly activated nodes.
        Options ['vaccination', 'transfer', 'decay', 'hybrid']
    :param activators: dict of node and list of activators for each node
    return the nodes that were healed in order to remove them from negative seed set if they exist in there
    """

    healed = list()
    for v in activators.keys():
        # the best aggression modeling configuration was cumulative
        summary = 0
        for u in activators[v]:
            summary += graph.nodes[u]['aggression_score']

        v_score = 0
        for u in activators[v]:
            # increase by w_i * A_i where w_i = A_i/sum. Thus, A_i^2/sum.
            v_score += power(graph.nodes[u]['aggression_score'], 2) / summary if summary != 0 else 0

        # actual healing
        if healing == 'vaccination':
            new_agg_score = 0
        elif healing == 'transfer':
            new_agg_score = v_score
        elif healing == 'decay':
            # e.g. if agg score = 0.5 and hops = 2, transfer 0.5 - 1/2*0.5 = 0.25
            # it heals but not that much as vaccination
            decay = 1/hops
            new_agg_score = v_score - (decay * v_score)
        else:  # hybrid
            coin_toss = rn.random()
            if coin_toss <= graph.nodes[v]['aggression_score']:
                new_agg_score = 0
            else:
                decay = 1 / hops
                new_agg_score = v_score - (decay * v_score)

        graph.nodes[v]['aggression_score'] = new_agg_score
        if new_agg_score == 0:
            healed.append(v)

    return healed


def transfer_aggression(graph, activation, activators):
    """
    Apply aggression transfer strategy during a diffusion process
    :param graph: the network
    :param activation: how to transfer aggression to newly activated nodes. Options ['random', 'top', 'cumulative']
    :param activators: dict of node and list of activators for each node
    """
    # adjust aggression scores of activated nodes
    if activation == 'random':
        for v in activators.keys():
            random_pick = random.choice(activators[v])
            graph.nodes[v]['aggression_score'] = graph.nodes[random_pick]['aggression_score']

    elif activation == 'top':
        for v in activators.keys():
            sorted_list = sorted(activators[v], reverse=True)
            top_pick = sorted_list[0]
            graph.nodes[v]['aggression_score'] = graph.nodes[top_pick]['aggression_score']

    elif activation == 'cumulative':
        for v in activators.keys():
            summary = 0
            for u in activators[v]:
                summary += graph.nodes[u]['aggression_score']

            v_score = 0
            for u in activators[v]:
                # increase by w_i * A_i where w_i = A_i/sum. Thus, A_i^2/sum.
                v_score += power(graph.nodes[u]['aggression_score'], 2) / summary

            graph.nodes[v]['aggression_score'] = v_score
    else:
        pass
    return


def get_network_aggression(g):
    """
    Calculates the aggression score of the whole network
    :param g: the network
    """

    agg_scores = nx.get_node_attributes(g, 'aggression_score')
    agg_scores = agg_scores.values()

    return sum(agg_scores)


def insert_aggression(g, filepath='../data/metrics.predictions.csv'):
    df = pd.read_csv(filepath)
    df['reverse_prediction'] = 1 - df['prediction']  # revert aggression scores as scores close to 0 showed more aggressive users
    node_levels = dict(zip(df['userid'], df['reverse_prediction']))
    nx.set_node_attributes(g, node_levels, 'aggression_score')

    # keep only nodes with aggression score
    nodes = [x for x, y in g.nodes(data=True) if 'aggression_score' in y]
    graph = g.subgraph(nodes)

    return graph


def create_config_string(row=None, args=None, blocking=False):
    """
    Creates the configuration string that will be used as x axis item
    :param blocking: if True then this experiment is about blocking minimization
    :param args: the configuration parameters
    :param row: the experiment
    :return: configuration string
    """
    model = row['Diffusion Model'] if not args else args.model
    config = model

    graph = row['Graph'] if not args else args.graph
    if 'power' in graph:
        config = '_'.join([config, 'P'])
    elif 'jaccard' in graph:
        config = '_'.join([config, 'J'])
    elif 'weighted' in graph:
        config = '_'.join([config, 'W'])
    else:
        config = '_'.join([config, 'R'])

    if blocking:
        seed_strategy = row['Block Type'] if not args else args.strategy
    else:
        seed_strategy = row['Seed Strategy'] if not args else args.strategy
    config = '_'.join([config, seed_strategy])

    if model == 'ic':
        activation = row['Activation Strategy'] if not args else args.activation

        if activation == 'random':
            config = '_'.join([config, 'r'])
        elif activation == 'top':
            config = '_'.join([config, 't'])
        else:
            config = '_'.join([config, 'c'])
    else:
        threshold = row['Threshold Strategy'] if not args else args.threshold

        if threshold == 'aggression':
            config = '_'.join([config, 'a'])
        elif threshold == 'power':
            config = '_'.join([config, 'p'])
        else:
            config = '_'.join([config, 'r'])
    if blocking:
        if args:
            config = '_'.join([config, args.adjacency])
        else:
            config = '_'.join([config, row['Adjacency Type']])
    return config


def create_config_string_min_competitive(row=None, args=None):
    """
    Creates the configuration string that will be used as x axis item for minimization experiments
    :param args: the configuration parameters
    :param row: the experiment
    :return: configuration string
    """

    config = row['Model'] if not args else args.model

    graph = row['Graph'] if not args else args.graph

    if 'power' in graph:
        config = '_'.join([config, 'P'])
    elif 'jaccard' in graph:
        config = '_'.join([config, 'J'])
    elif 'weighted' in graph:
        config = '_'.join([config, 'W'])
    else:
        config = '_'.join([config, 'R'])


    seed_strategy = row['Seed Strategy'] if not args else args.strategy
    config = '_'.join([config, seed_strategy])

    healing_strategy = row['Healing Strategy'] if not args else args.healing
    if healing_strategy == 'vaccination':
        config = '_'.join([config, 'v'])
    elif healing_strategy == 'transfer':
        config = '_'.join([config, 't'])
    elif healing_strategy == 'decay':
        config = '_'.join([config, 'd'])
    else:
        config = '_'.join([config, 'h'])

    return config


def write_to_csv_modeling(config, activated, i_agg, f_agg, total_time):

    directory = 'results/modeling/{}'.format(config.model)

    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = '{}/total.csv'.format(directory)
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        headers = ['Graph', 'Seed Size', 'Seed Strategy', 'Diffusion Model',
                   'Activation Strategy', 'Threshold Strategy', 'Activated Nodes',
                   'Initial Aggression Score', 'Final Aggression Score', 'Total Time']
        writer = csv.DictWriter(file, delimiter=',', fieldnames=headers)

        if not file_exists:
            writer.writeheader()
        if config.model == 'ic':
            writer.writerow({'Graph': config.graph,
                             'Seed Size': config.seedsize,
                             'Seed Strategy': config.strategy,
                             'Diffusion Model': config.model,
                             'Activation Strategy': config.activation,
                             'Threshold Strategy': '-',
                             'Activated Nodes': activated,
                             'Initial Aggression Score': i_agg,
                             'Final Aggression Score': f_agg,
                             'Total Time': total_time})
        else:
            writer.writerow({'Graph': config.graph,
                             'Seed Size': config.seedsize,
                             'Seed Strategy': config.strategy,
                             'Diffusion Model': config.model,
                             'Activation Strategy': '-',
                             'Threshold Strategy': config.threshold,
                             'Activated Nodes': activated,
                             'Initial Aggression Score': i_agg,
                             'Final Aggression Score': f_agg,
                             'Total Time': total_time})

    return


def write_to_csv_min_competitive(config, n_activated, p_activated, i_agg, f_agg, total_time):
    filename = 'results/competitive/total.csv'
    file_exists = os.path.isfile(filename)

    if not os.path.exists('results/competitive'):
        os.makedirs('results/competitive')

    with open(filename, mode='a', newline='') as file:
        headers = ['Graph', 'Seed Size', 'Seed Strategy', 'Model', 'Healing Strategy', 'Negative Activated Nodes',
                   'Positive Activated Nodes', 'Initial Aggression Score', 'Final Aggression Score', 'Total Time']
        writer = csv.DictWriter(file, delimiter=',', fieldnames=headers)

        if not file_exists:
            writer.writeheader()

        writer.writerow({'Graph': config.graph,
                         'Seed Size': config.seedsize,
                         'Seed Strategy': config.strategy,
                         'Model': config.model,
                         'Healing Strategy': config.healing,
                         'Negative Activated Nodes': n_activated,
                         'Positive Activated Nodes': p_activated,
                         'Initial Aggression Score': i_agg,
                         'Final Aggression Score': f_agg,
                         'Total Time': total_time})

    return


def write_to_csv_min_blocking(config, activated, i_agg, f_agg, total_time):
    filename = 'results/blocking/total.csv'
    file_exists = os.path.isfile(filename)

    if not os.path.exists('results/blocking'):
        os.makedirs('results/blocking')

    with open(filename, mode='a', newline='') as file:
        headers = ['Graph', 'Seed Size', 'Block Type', 'Adjacency Type', 'Seed Strategy', 'Diffusion Model',
                   'Activation Strategy', 'Threshold Strategy', 'Activated Nodes', 'Initial Aggression Score',
                   'Final Aggression Score', 'Total Time']
        writer = csv.DictWriter(file, delimiter=',', fieldnames=headers)

        if not file_exists:
            writer.writeheader()

        if config.model == 'ic':
            writer.writerow({'Graph': config.graph,
                             'Seed Size': config.seedsize,
                             'Block Type': config.strategy,
                             'Adjacency Type': config.adjacency,
                             'Seed Strategy': config.seedstrategy,
                             'Diffusion Model': config.model,
                             'Activation Strategy': config.activation,
                             'Threshold Strategy': '-',
                             'Activated Nodes': activated,
                             'Initial Aggression Score': i_agg,
                             'Final Aggression Score': f_agg,
                             'Total Time': total_time})
        else:
            writer.writerow({'Graph': config.graph,
                             'Seed Size': config.seedsize,
                             'Block Type': config.strategy,
                             'Adjacency Type': config.adjacency,
                             'Seed Strategy': config.seedstrategy,
                             'Diffusion Model': config.model,
                             'Activation Strategy': '-',
                             'Threshold Strategy': config.threshold,
                             'Activated Nodes': activated,
                             'Initial Aggression Score': i_agg,
                             'Final Aggression Score': f_agg,
                             'Total Time': total_time})

    return


def write_metrics(config, repeat, cosines, pearsons, spearmans, kendalls, agg_score_threshold, exp_type='modeling', seed_size=5594):
    """
    Writes three lines in a csv. First is about cosine similarity, second pearson R and third spearman R
    :param seed_size: the size of the initial seed set
    :param exp_type: type of experiment
    :param agg_score_threshold: aggression score threshold
    :param config: the configuration
    :param repeat: the repeat of the experiment
    :param cosines: list of cosine similarities
    :param pearsons: list of pearson similarities
    :param spearmans: list of spearman similarities
    :param kendalls: list of kendall similarities
    """
    filename = 'snapshots/{}/{}/{}/{}/total_{}.csv'.format(exp_type, seed_size, config, repeat, agg_score_threshold)
    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, mode='a', newline='') as file:
        wr = csv.writer(file)

        new_cosines = ['cosine']
        new_cosines.extend(cosines)
        new_pearsons = ['pearson']
        new_pearsons.extend(pearsons)
        new_spearmans = ['spearman']
        new_spearmans.extend(spearmans)
        new_kendalls = ['kendall']
        new_kendalls.extend(kendalls)

        wr.writerow(new_cosines)
        wr.writerow(new_pearsons)
        wr.writerow(new_spearmans)
        wr.writerow(new_kendalls)


def write_aggressions(config, repeat, aggressions, exp_type='minimization', seed_size=5594):
    """
    Writes one line in a csv. It refers to aggression scores per snapshot.
    :param exp_type: type of experiment
    :param config: the configuration
    :param repeat: the repeat of the experiment
    :param aggressions: a list of aggression scores per snapshot
    :param seed_size: the size of the initial seed set
    """
    filename = 'snapshots/{}/{}/{}/{}/total_aggressions.csv'.format(exp_type, seed_size, config, repeat)

    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, mode='a', newline='') as file:
        wr = csv.writer(file)

        new_aggressions = ['aggression']
        new_aggressions.extend(aggressions)
        wr.writerow(new_aggressions)


def anova_table(aov):
    """
    Calculates the effect sizes r square, mean square and omega square
    :param aov: the anova table
    """
    aov['mean_sq'] = aov[:]['sum_sq'] / aov[:]['df']

    aov['eta_sq'] = aov[:-1]['sum_sq'] / sum(aov['sum_sq'])

    aov['omega_sq'] = (aov[:-1]['sum_sq'] - (aov[:-1]['df'] * aov['mean_sq'][-1])) / (
            sum(aov['sum_sq']) + aov['mean_sq'][-1])

    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov[cols]
    return aov