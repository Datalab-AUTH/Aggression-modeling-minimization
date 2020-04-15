import networkx as nx
from numpy.random import rand
from numpy import max, min


def jaccard_overlap(graph, inplace=True):
    """
    Calculates the jaccard overlap of sets of neighbors for each edge
    :param graph: the network
    :param inplace: if True then jaccard overlap is used as weight else it is a separate attribute
    :return: the jaccard overlap in [0,1]
    """
    attr = 'weight' if inplace else 'jaccard'

    for edge in graph.edges():
        Ni = graph.neighbors(edge[0])
        Nj = graph.neighbors(edge[1])

        s1 = set(Ni)
        s2 = set(Nj)
        jaccard_similarity = float(len(s1.intersection(s2))) / len(s1.union(s2))

        graph[edge[0]][edge[1]][attr] = jaccard_similarity

    normalize(graph, attr)


def node_power_scores(graph):
    """
    Calculates the normalized power score of every node in the network
    A node with no out neighbors will have:
        power score = 0 if it hasn't got any in neighbors (marked as -2)
        power score = 1 if it has some in neighbors (marked as -1)
    """
    power_scores = dict()

    for node in graph.nodes():
        in_degree = graph.in_degree(node)
        out_degree = graph.out_degree(node)

        if out_degree == 0:  # mark the singularity node for the normalization step
            if in_degree != 0:
                power_scores[node] = -1
            else:
                power_scores[node] = -2
        else:
            power_scores[node] = in_degree / out_degree

    # normalize to [0-1]
    for node in graph.nodes():
        if power_scores[node] == -1:  # it has in neighbors but no out neighbors
            power_scores[node] = 1
        elif power_scores[node] == -2:  # it has neither in neighbors nor out neighbors
            power_scores[node] = 0
        else:
            max_score = -1
            for neighbor in graph.neighbors(node):
                if power_scores[neighbor] > max_score:
                    max_score = power_scores[neighbor]
            if power_scores[node] > max_score:
                max_score = power_scores[node]
            power_scores[node] = float(power_scores[node]) / max_score

    return power_scores


def power_score(graph, power_type='frac', inplace=True):
    """
    Calculates the power score of an edge as the power score of the source, the dest node or a fraction of those two
    At time step t, with regard to a diffusion process,
    source is the node activated at t-1 and dest the one that might be activated at t
    :param graph: the network
    :param power_type: the type of the edge's power score. One of [source, dest, frac]
    :param inplace: if True then power score is used as weight else it is a separate attribute
    :return: power score normalized in [0,1]
    """

    attr = 'weight' if inplace else 'power'

    if power_type not in ['source', 'dest', 'frac']:
        print("Power type should be one of ['source', 'dest', 'frac']")
        return

    # calculate node power scores
    power_scores = node_power_scores(graph)

    for edge in graph.edges():
        if power_type == 'source':
            graph[edge[0]][edge[1]][attr] = power_scores[edge[0]]
        elif power_type == 'dest':
            graph[edge[0]][edge[1]][attr] = power_scores[edge[1]]
        else:  # fraction. It needs normalization
            src_power = power_scores[edge[0]]
            dest_power = power_scores[edge[1]]
            if src_power == 0:  # mark the singularity node for the normalization step
                if dest_power != 0:
                    graph[edge[0]][edge[1]][attr] = -1
                else:
                    graph[edge[0]][edge[1]][attr] = -2
            else:
                graph[edge[0]][edge[1]][attr] = dest_power / src_power

    normalize(graph, attr)


def weighted_overlap(graph, power_type='frac', inplace=True):
    """
    A method combining both power score and jaccard overlap into a single weight
    :param inplace: if True then power score is used as weight else it is a separate attribute
    :param graph: the network
    :param power_type: the type of the edge's power score. One of [source, dest, frac]
    :return: weighted_overlap in [0,1]
    """
    jaccard_overlap(graph, inplace=False)
    power_score(graph, power_type, inplace=False)

    attr = 'weight' if inplace else 'weighted_overlap'

    for edge in graph.edges():
        graph[edge[0]][edge[1]][attr] = graph[edge[0]][edge[1]]['power'] * graph[edge[0]][edge[1]]['jaccard']

    normalize(graph, attr)

    # remove unnecessary attributes
    for u, v, attributes in graph.edges(data=True):
        for attr in ['power', 'jaccard']:
            if attr in attributes:
                del attributes[attr]


def normalize(graph, attr):
    """
    normalize to [0-1]
    """
    edge_scores = nx.get_edge_attributes(graph, attr)
    max_score = max(list(edge_scores.values()))
    min_score = min(list(edge_scores.values()))
    for edge in graph.edges():
        if graph[edge[0]][edge[1]][attr] == -1:
            graph[edge[0]][edge[1]][attr] = 1
        elif graph[edge[0]][edge[1]][attr] == -2:
            graph[edge[0]][edge[1]][attr] = 0
        else:
            graph[edge[0]][edge[1]][attr] = (graph[edge[0]][edge[1]][attr] - min_score) / (max_score - min_score)


def random(graph):
    """
    Apply random weights
    """
    weights = list(rand(len(graph.edges)))

    for idx, edge in enumerate(graph.edges):
        graph[edge[0]][edge[1]]['weight'] = weights[idx]
