import networkx as nx
import time
import pandas as pd
from seed_selection import single_discount
from copy import deepcopy
import utility
import random
import os
import csv
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm

def insert_aggression(g, filepath='../data/smallnetwork/labels.txt'):
    df = pd.read_csv(filepath, sep='\t', header=None)
    node_levels = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

    nx.set_node_attributes(g, node_levels, 'aggression_score')

    # keep only nodes with aggression score
    nodes = [x for x, y in g.nodes(data=True) if 'aggression_score' in y]
    graph = g.subgraph(nodes)

    return graph


def take_snapshot(graph, model, step):

    if model == 'ic':
        config = 'small_ic'
    else:
        config = 'small_lt'


    directory = 'snapshots/small/{}'.format(config)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open('{}/{}.csv'.format(directory, step), mode='a', newline='') as file:
        for node, agg_score in graph.nodes(data='aggression_score'):
            wr = csv.writer(file)
            wr.writerow([node, agg_score])


def run_ic(graph, s):

    print('IC Model')

    start = time.time()

    agg_scores = list()

    seeds = deepcopy(s)  # contains all the currently activated nodes
    activations = deepcopy(s)   # keeps track of the new activations at each step
    print("Activated nodes: ", (len(seeds)))


    take_snapshot(graph, 'ic', 0)

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

        utility.transfer_aggression(graph, 'cumulative', activators)

        print("Activated nodes: ", (len(seeds)))

        # Aggression score
        agg_score = utility.get_network_aggression(graph)
        print('Aggression: ', agg_score)
        agg_scores.append(agg_score)


        take_snapshot(graph, 'ic', iteration)

        iteration += 1

    print()
    print('IC time:{}'.format(time.time() - start))
    print('Total activated nodes: ', (len(seeds)))
    print('Total aggression:', agg_scores[-1])

    return len(seeds), agg_scores


def run_lt(graph, s):

    print('LT Model')

    start = time.time()

    thresholds = dict()
    for node in graph.nodes:
        thresholds[node] = graph.nodes[node]['aggression_score']

    agg_scores = list()

    seeds = deepcopy(s)
    activations = deepcopy(s)
    print("Activated nodes: ", (len(seeds)))

    activators = dict()
    node_influences = dict(zip(graph.nodes, [0] * len(graph.nodes)))


    take_snapshot(graph, 'lt', 0)

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

        take_snapshot(graph, 'lt', iteration)
        iteration += 1

    print()
    print("LT time:{}".format(time.time() - start))
    print('Total activated nodes: ', (len(seeds)))
    print('Total aggression:', agg_scores[-1])

    return len(seeds), agg_scores


def run_experiment(g: nx.DiGraph, model):
    # Attach aggression related attributes to nodes
    g = insert_aggression(g)
    seeds = single_discount(g, 5594)

    if model == 'ic':
        activated, agg_scores = run_ic(g, seeds)
    else:  # lt
        activated, agg_scores = run_lt(g, seeds)

    return activated,  agg_scores


if __name__ == '__main__':
    # g = nx.read_gpickle("data/graphs/small_jaccard")
    # start = time.time()
    #
    # activated, agg_scores = run_experiment(g, 'ic')
    # total_time = time.time() - start
    # print("Total time: ", total_time)
    # print()

    df = pd.read_csv('../data/smallnetwork/labels.txt', sep='\t', header=None)
    y_true = df.iloc[:, 2]
    y_true = [y if y != -1 else 0 for y in y_true]

    positives = len([y for y in y_true if y == 1])
    negatives = len([y for y in y_true if y == 0])
    df_pred = pd.read_csv('snapshots/small/total.csv')

    col_scores = dict()
    for (columnName, columnData) in tqdm(df_pred.iteritems()):
        scores = list()
        # for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        #     data = columnData
        #     if columnName == 'node':
        #         continue
        #
        #     data = [0 if x <= threshold else 1 for x in data]
        #
        #     score = roc_auc_score(y_true, data)
        #     scores.append(score)

        col_scores.update({columnName: roc_auc_score(y_true, columnData)})

    print(col_scores)

    # for (columnName, columnData) in tqdm(df_pred.iteritems()):
    #     fpr, tpr, ts = roc_curve(y_true, columnData)
    #     plt.title(str(columnName) + ': ' + str(ts))
    #     plt.plot(fpr, tpr)
    #     plt.plot([0,1], [0,1], 'k--')
    #     plt.ylabel('TPR')
    #     plt.xlabel('FPR')
    #     plt.show()

        # if columnName == 'node':
        #     continue
        # columnData = [0 if x <= 0.4 else 1 for x in columnData]
        # print('Column Name : ', columnName)
        # macro_prec = precision_score(y_true, columnData, average='macro')
        # macro_rec = recall_score(y_true, columnData, average='macro')
        # macro_f1 = f1_score(y_true, columnData, average='macro')
        # micro_prec = precision_score(y_true, columnData, average='micro')
        # micro_rec = recall_score(y_true, columnData, average='micro')
        # micro_f1 = f1_score(y_true, columnData, average='micro')
        #
        # print('Macro Scores')
        # print('Precision: {}, Recall: {}, F1: {}'.format(macro_prec, macro_rec, macro_f1))
        #
        # print('Micro Scores')
        # print('Precision: {}, Recall: {}, F1: {}'.format(micro_prec, micro_rec, micro_f1))