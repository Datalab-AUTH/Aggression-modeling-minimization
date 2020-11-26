import networkx as nx
import pandas as pd
from scipy import stats
from numpy import dot
from tqdm import tqdm
from numpy.linalg import norm
import csv
import utility
import os
import glob


def calc_metrics(args, exp_type='modeling'):
    agg_score_threshold = args.aggression_threshold
    metric_type = args.metric_type
    configuration = args.configuration
    seedsize = args.seedsize

    # the different configurations
    folders = [f for f in glob.glob('snapshots/{}/**'.format(exp_type), recursive=False)]

    if seedsize: # if a specific seed size is given then search only for this one
        seed_sizes = [seedsize]
    else:
        seed_sizes = [f.split('\\')[1] for f in folders]

    for seed_size in seed_sizes:
        if configuration: # if a specific configuration is given then search only for this one
            configs = [configuration]
        else:
            configs = [f.split('\\')[1] for f in glob.glob('snapshots/{}/{}/**'.format(exp_type, seed_size), recursive=False)]

        repeats = dict()
        for config in configs:
            folders = [f for f in glob.glob('snapshots/{}/{}/{}/**'.format(exp_type, seed_size, config), recursive=False)]
            repeats.update({config: len(folders)})

        trace = dict()  # (config: {repeat_i: num_of_steps_in_repeat_i, ...}) items
        for config in configs:
            steps_per_repeat = dict()
            for i in range(repeats[config]):
                step_files = [f for f in glob.glob('snapshots/{}/{}/{}/{}/*.csv'.format(exp_type, seed_size, config, i), recursive=False)
                              if 'total' not in f]
                steps_per_repeat.update({i: len(step_files)})
            trace.update({config: steps_per_repeat})

        for config in tqdm(trace.keys()):  # eg. 'ic_J_r_r'
            if '_J_' in config:
                graph = nx.read_gpickle('data/graphs/Twitter_jaccard')
            elif '_P_' in config:
                graph = nx.read_gpickle('data/graphs/Twitter_power')
            elif '_W_' in config:
                graph = nx.read_gpickle('data/graphs/Twitter_weighted_overlap')
            else:
                graph = nx.read_gpickle('data/graphs/Twitter_random')
            graph = utility.insert_aggression(graph)

            repeats = trace[config].keys()
            if metric_type == 'similarity':
                for repeat in tqdm(repeats):  # eg 0
                    cosines = list()
                    pearsons = list()
                    spearmans = list()
                    kendalls = list()

                    steps = trace[config][repeat]
                    for step in range(1, steps):  # First step is skipped because validation vector is calculated on intervals
                        prev_snapshot = '{}/{}/{}/{}.csv'.format(seed_size, config, repeat, step-1)
                        current_snapshot = '{}/{}/{}/{}.csv'.format(seed_size, config, repeat, step)

                        val_vector = calc_validation_vector(graph, prev_snapshot, current_snapshot, agg_score_threshold, exp_type)
                        cosine, pearson, spearman, kendall = calc_metrics_for_vector('../data/vectors/vectors_new', val_vector)

                        cosines.append(cosine)
                        pearsons.append(pearson)
                        spearmans.append(spearman)
                        kendalls.append(kendall)

                    utility.write_metrics(config, repeat, cosines, pearsons, spearmans, kendalls, agg_score_threshold, exp_type, seed_size)
            elif metric_type == 'aggression':
                for repeat in tqdm(repeats):
                    aggression_scores = list()
                    steps = trace[config][repeat]
                    for step in range(steps):
                        snapshot = '{}/{}/{}/{}.csv'.format(seed_size, config, repeat, step)
                        data = pd.read_csv('snapshots/{}/{}'.format(exp_type, snapshot), names=['User', 'Aggression Score'])
                        aggression_score = data['Aggression Score'].sum()
                        aggression_scores.append(aggression_score)
                    utility.write_aggressions(config, repeat, aggression_scores, exp_type, seed_size)
            else:
                return


def find_aggressive(snapshot, agg_score_threshold, exp_type):
    """
    Find aggressive users based on a specific snapshot and a specific aggression score threshold.
    Set the corresponding attribute for every graph node
    """

    # read data
    data = pd.read_csv('snapshots/{}/{}'.format(exp_type, snapshot), names=['User', 'Aggression Score'])
    # determine which users are aggressive
    data['Aggressive'] = data.apply(lambda row: row['Aggression Score'] >= agg_score_threshold, axis=1)
    user_agg = dict()

    # classify users to aggressive or normal based on the scores of the given snapshot
    for idx, row in data.iterrows():
        user_agg.update({row['User']: row['Aggressive']})

    return user_agg


def calc_snapshot_vector(graph: nx.DiGraph, snapshot: str, agg_score_threshold: float, normalized=True, data=False, exp_type='modeling'):
    """
    Calculate node and edge metrics given a network snapshot
    :param data: whether to return snapshot graph or not
    :param normalized: whether to normalize values or not
    :param agg_score_threshold: the threshold to find aggressive users
    :param snapshot: the snapshot we are interested in
    :param graph: the generic graph
    """

    results = dict()

    user_dict = find_aggressive(snapshot, agg_score_threshold, exp_type)

    # node metrics
    nodes = graph.nodes()
    # for node in graph.nodes(data=True):
    #     print(node)

    normal = len([user for user in user_dict.keys() if not user_dict[user]])
    aggressive = len([user for user in user_dict.keys() if user_dict[user]])

    results['n'] = normal / len(nodes) if normalized else normal
    results['a'] = aggressive / len(nodes) if normalized else aggressive

    # edge metrics
    edges = graph.edges()
    normal_normal = 0
    normal_aggressive = 0
    aggressive_normal = 0
    aggressive_aggressive = 0
    for edge in edges:
        u = edge[0]
        v = edge[1]
        u_aggressive = user_dict[u]
        v_aggressive = user_dict[v]
        if not u_aggressive and not v_aggressive:
            normal_normal = normal_normal + 1
        if not u_aggressive and v_aggressive:
            normal_aggressive = normal_aggressive + 1
        if u_aggressive and not v_aggressive:
            aggressive_normal = aggressive_normal + 1
        if u_aggressive and v_aggressive:
            aggressive_aggressive = aggressive_aggressive + 1

    normal_normal = normal_normal
    normal_aggressive = normal_aggressive
    aggressive_normal = aggressive_normal
    aggressive_aggressive = aggressive_aggressive
    results['N-N'] = normal_normal / len(edges) if normalized else normal_normal
    results['N-A'] = normal_aggressive / len(edges) if normalized else normal_aggressive
    results['A-N'] = aggressive_normal / len(edges) if normalized else aggressive_normal
    results['A-A'] = aggressive_aggressive / len(edges) if normalized else aggressive_aggressive

    if data:
        return results, user_dict
    else:
        return results


def calc_validation_vector(graph: nx.DiGraph, previous: str, current: str, agg_score_threshold: float, exp_type: str = 'modeling'):
    """
    Given 2 snapshot graphs statuses it calculates the final validation vector
    """

    i_snapshot_vector, i_user_dict = calc_snapshot_vector(graph, previous, agg_score_threshold, normalized=False, data=True, exp_type=exp_type)
    f_snapshot_vector, f_user_dict = calc_snapshot_vector(graph, current, agg_score_threshold, data=True, exp_type=exp_type)

    user_dict = dict()  # (node: (initial_agg_state, final_agg_state)) items
    for user in i_user_dict.keys():
        user_dict.update({user: (i_user_dict[user], f_user_dict[user])})

    results = f_snapshot_vector

    n_n = 0
    n_a = 0
    a_n = 0
    a_a = 0
    for u in user_dict:
        n_n += 1 if (not user_dict[u][0] and not user_dict[u][1]) else 0
        n_a += 1 if (not user_dict[u][0]and user_dict[u][1]) else 0
        a_n += 1 if (user_dict[u][0] and not user_dict[u][1]) else 0
        a_a += 1 if (user_dict[u][0] and user_dict[u][1]) else 0

    # n->n means: normal users who remained normal. n->a means: normal users who became aggressive
    results['n->n'] = n_n / i_snapshot_vector['n'] if i_snapshot_vector['n'] != 0 else 0
    results['n->a'] = n_a / i_snapshot_vector['n'] if i_snapshot_vector['n'] != 0 else 0
    results['a->n'] = a_n / i_snapshot_vector['a'] if i_snapshot_vector['a'] != 0 else 0
    results['a->a'] = a_a / i_snapshot_vector['a'] if i_snapshot_vector['a'] != 0 else 0

    # edge metrics
    edges = graph.edges()
    nn_nn = 0
    nn_na = 0
    nn_an = 0
    nn_aa = 0

    na_nn = 0
    na_na = 0
    na_an = 0
    na_aa = 0

    an_nn = 0
    an_na = 0
    an_an = 0
    an_aa = 0

    aa_nn = 0
    aa_na = 0
    aa_an = 0
    aa_aa = 0

    for edge in edges:
        u = edge[0]
        v = edge[1]
        if not user_dict[u][0] and not user_dict[v][0]:  # N-N
            nn_nn += 1 if not user_dict[u][1] and not user_dict[v][1] else 0  # N-N -> N-N
            nn_na += 1 if not user_dict[u][1] and user_dict[v][1] else 0  # N-N -> N-A
            nn_an += 1 if user_dict[u][1] and not user_dict[v][1] else 0  # N-N -> A-N
            nn_aa += 1 if user_dict[u][1] and user_dict[v][1] else 0  # N-N -> A-A
        if not user_dict[u][0] and user_dict[v][0]:  # N-A
            na_nn += 1 if not user_dict[u][1] and not user_dict[v][1] else 0
            na_na += 1 if not user_dict[u][1] and user_dict[v][1] else 0
            na_an += 1 if user_dict[u][1] and not user_dict[v][1] else 0
            na_aa += 1 if user_dict[u][1] and user_dict[v][1] else 0
        if user_dict[u][0] and not user_dict[v][0]:  # A-N
            an_nn += 1 if not user_dict[u][1] and not user_dict[v][1] else 0
            an_na += 1 if not user_dict[u][1] and user_dict[v][1] else 0
            an_an += 1 if user_dict[u][1] and not user_dict[v][1] else 0
            an_aa += 1 if user_dict[u][1] and user_dict[v][1] else 0
        if user_dict[u][0] and user_dict[v][0]:  # A-A
            aa_nn += 1 if not user_dict[u][1] and not user_dict[v][1] else 0
            aa_na += 1 if not user_dict[u][1] and user_dict[v][1] else 0
            aa_an += 1 if user_dict[u][1] and not user_dict[v][1] else 0
            aa_aa += 1 if user_dict[u][1] and user_dict[v][1] else 0

    results['N-N->N-N'] = nn_nn / i_snapshot_vector['N-N'] if i_snapshot_vector['N-N'] != 0 else 0
    results['N-N->N-A'] = nn_na / i_snapshot_vector['N-N'] if i_snapshot_vector['N-N'] != 0 else 0
    results['N-N->A-N'] = nn_an / i_snapshot_vector['N-N'] if i_snapshot_vector['N-N'] != 0 else 0
    results['N-N->A-A'] = nn_aa / i_snapshot_vector['N-N'] if i_snapshot_vector['N-N'] != 0 else 0

    results['N-A->N-N'] = na_nn / i_snapshot_vector['N-A'] if i_snapshot_vector['N-A'] != 0 else 0
    results['N-A->N-A'] = na_na / i_snapshot_vector['N-A'] if i_snapshot_vector['N-A'] != 0 else 0
    results['N-A->A-N'] = na_an / i_snapshot_vector['N-A'] if i_snapshot_vector['N-A'] != 0 else 0
    results['N-A->A-A'] = na_aa / i_snapshot_vector['N-A'] if i_snapshot_vector['N-A'] != 0 else 0

    results['A-N->N-N'] = an_nn / i_snapshot_vector['A-N'] if i_snapshot_vector['A-N'] != 0 else 0
    results['A-N->N-A'] = an_na / i_snapshot_vector['A-N'] if i_snapshot_vector['A-N'] != 0 else 0
    results['A-N->A-N'] = an_an / i_snapshot_vector['A-N'] if i_snapshot_vector['A-N'] != 0 else 0
    results['A-N->A-A'] = an_aa / i_snapshot_vector['A-N'] if i_snapshot_vector['A-N'] != 0 else 0

    results['A-A->N-N'] = aa_nn / i_snapshot_vector['A-A'] if i_snapshot_vector['A-A'] != 0 else 0
    results['A-A->N-A'] = aa_na / i_snapshot_vector['A-A'] if i_snapshot_vector['A-A'] != 0 else 0
    results['A-A->A-N'] = aa_an / i_snapshot_vector['A-A'] if i_snapshot_vector['A-A'] != 0 else 0
    results['A-A->A-A'] = aa_aa / i_snapshot_vector['A-A'] if i_snapshot_vector['A-A'] != 0 else 0

    # create validation vector from dict in the right order
    val_vector = [results['n'], results['a'],
                  results['N-N'], results['N-A'], results['A-N'], results['A-A'],
                  results['n->n'], results['n->a'], results['a->n'], results['a->a'],
                  results['N-N->N-N'], results['N-N->N-A'], results['N-N->A-N'], results['N-N->A-A'],
                  results['N-A->N-N'], results['N-A->N-A'], results['N-A->A-N'], results['N-A->A-A'],
                  results['A-N->N-N'], results['A-N->N-A'], results['A-N->A-N'], results['A-N->A-A'],
                  results['A-A->N-N'], results['A-A->N-A'], results['A-A->A-N'], results['A-A->A-A']
                  ]

    return val_vector


def take_snapshot(graph, args, instance, step, exp_type='modeling'):
    """
    Write node, aggression score pairs in file for a given snapshot
    :param exp_type: type of experiment
    :param instance: current repeat of the experiment
    :param graph: the network
    :param args: the experiment setup
    :param step: a number defining the snapshot. E.g. snapshot 0
    """
    if exp_type == 'modeling':
        config = utility.create_config_string(args=args)
    elif exp_type == 'minimization':
        config = utility.create_config_string_min_competitive(args=args)
    elif exp_type == 'blocking':
        config = utility.create_config_string(args=args, blocking=True)
    else:
        return

    directory = 'snapshots/{}/{}/{}/{}'.format(exp_type, args.seedsize, config, instance)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open('{}/{}.csv'.format(directory, step), mode='a', newline='') as file:
        for node, agg_score in graph.nodes(data='aggression_score'):
            wr = csv.writer(file)
            wr.writerow([node, agg_score])


def calc_metrics_for_vector(ground_truth_file, val_vector):

    with open(ground_truth_file, 'r') as f:
        lines = f.readlines()
        vector_str = lines[-1]
        vector_str = vector_str[1:-1]
        vector = [float(x.strip()) for x in vector_str.split(',')]  # split in comma, remove whitespace,turn into float

        # indices = [0,2,6,10] # remove sensitive elements from the proposed validation vector
        # val_vector = [x for i, x in enumerate(val_vector) if i not in indices]
        # vector = [x for i, x in enumerate(vector) if i not in indices]

        cosine_sim = dot(val_vector, vector) / (norm(vector)*norm(val_vector))
        pearson = stats.pearsonr(val_vector, vector)[0]  # high pearson (+1 or -1) means linear correlation
        spearman = stats.spearmanr(val_vector, vector)[0]  # high spearman (+1 or -1) means monotonic correlation
        kendall, p_value = stats.kendalltau(val_vector, vector)
        return cosine_sim, pearson, spearman, kendall
