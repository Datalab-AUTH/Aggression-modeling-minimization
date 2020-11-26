import argparse
import time
import networkx as nx
import seed_selection
import utility
from metrics import calc_metrics
from models.influence_maximization.ic import run_ic
from models.influence_maximization.lt import run_lt
from models.influence_minimization.netshield import net_shield
from models.influence_minimization.netmelt import net_melt
import pickle
from tqdm import tqdm


def run_experiment(g: nx.DiGraph, args, instance: int):
    """
    The actual experiment
    Values for negative cascade are hard coded. They represent the best experiment of aggression modeling phase
    :param instance: current repeat of the experiment
    :param args: command-line arguments
    :param g: the graph
    """

    # Attach aggression related attributes to nodes
    g = nx.DiGraph(utility.insert_aggression(g))  # create a new graph as the old one freezes

    if args.adjacency == 'aggression':
        # net shield will use an adjacency where weight of (u,v) pair is agg score u * agg score v
        graph = nx.DiGraph()
        for edge in tqdm(g.edges(data=True)):
            u = edge[0]
            v = edge[1]
            agg_u = g.nodes[u]['aggression_score']
            agg_v = g.nodes[v]['aggression_score']
            graph.add_edge(u, v, weight=(agg_u * agg_v))

        if args.strategy == 'node':
            nodes_to_remove = net_shield(graph, args.seedsize)
            g.remove_nodes_from(nodes_to_remove)
        else:
            edges_to_remove = net_melt(graph, args.seedsize)
            g.remove_edges_from(edges_to_remove)
    else:
        if args.strategy == 'node':
            nodes_to_remove = net_shield(g, args.seedsize)
            g.remove_nodes_from(nodes_to_remove)
        else:
            edges_to_remove = net_melt(g, args.seedsize)
            g.remove_edges_from(edges_to_remove)

    # negative seeds are selected according to the best strategy of aggression modeling phase
    if args.seedstrategy == 'aa':
        negative_seeds = seed_selection.all_aggressive(g)
    elif args.seedstrategy == 'sd':
        negative_seeds = seed_selection.single_discount(g, args.seedsize)
    elif args.seedstrategy == 'dd':
        negative_seeds = seed_selection.degree_discount(g, args.seedsize)
    else:
        negative_seeds = seed_selection.random(g, args.seedsize)

    if args.model == 'ic':
        activated, agg_scores = run_ic(g, negative_seeds, args, instance, 'blocking')
    else:  # lt
        activated, agg_scores = run_lt(g, negative_seeds, args, instance, 'blocking')

    return activated, agg_scores


def experiment(args, save):
    """
    The wrapper experiment function
    """

    print('Experiment will be repeated {} times.'.format(args.num_experiments))
    print()

    # Network topology
    g = nx.read_gpickle(args.graph)

    # actual experiment
    for i in range(args.num_experiments):
        start = time.time()

        activated, agg_scores = run_experiment(g, args, i)

        total_time = time.time() - start
        print("Total time: ", total_time)
        print()

        if save:
            utility.write_to_csv_min_blocking(args, activated, agg_scores[0], agg_scores[-1], total_time)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Blocking Aggression Minimization simulation framework')
    subparsers = parser.add_subparsers(dest='mode', help="simulation or metric. Simulation runs the actual simulations"
                                                         " of the diffusion experiment and produces the intermediate snapshots."
                                                         " Metric uses the snapshots to calculate cosine similarity, Pearson R, Spearman R, Kendal Tau and aggression"
                                                         " needed for the various plots")

    simulation_parser = subparsers.add_parser('simulation', help="Simulation of a blocking aggression minimization process based on the given parameters. The negative cascade uses the best configuration found during the author's experimental process. If this needs to be changed, the code has to be altered")
    simulation_parser.add_argument("graph", type=str, help="Path to graph pickle file")
    simulation_parser.add_argument("snapshot", type=bool, help='If True, apart from the first and last step, it will create intermediate snapshots during the process')
    simulation_parser.add_argument("seedsize", type=int, help='The seed set size. That is, the number of nodes or edges that the blocking mechanism will remove. Also it is the number of initially infected nodes by the cascade. 5594 is the total number of aggressive users')
    simulation_parser.add_argument("strategy", type=str, choices=['node', 'edge'], help="Whether to remove nodes or edges. Node removal is more intrusive")
    simulation_parser.add_argument("adjacency", type=str, choices=['normal', 'aggression'], help="Type of adjacency. Normal adjacency leads to abusive node or edge detection, while aggression adjacency is fine-tuned towards aggression detection")
    simulation_parser.add_argument("seedstrategy", type=str, choices=['r', 'aa', 'sd', 'dd'], help="Seed node selection strategy. Short names for 'Random', 'All aggressive', 'Single Discount' and 'Degree Discount'")

    sim_subparsers = simulation_parser.add_subparsers(dest='model')

    ic_parser = sim_subparsers.add_parser('ic', help='Simulation of an aggression diffusion process based on IC model')
    ic_parser.add_argument('activation', type=str, choices=['random', 'top', 'cumulative'], help="Activation strategy during diffusion process")
    ic_parser.add_argument('num_experiments', type=int, help='How many times to repeat an experiment. > 1 is suggested as IC is probabilistic')

    lt_parser = sim_subparsers.add_parser('lt', help='Simulation of an aggression diffusion process based on LT model')
    lt_parser.add_argument('threshold', type=str, choices=['aggression', 'power'], help="Node threshold strategy")
    lt_parser.add_argument('num_experiments', type=int, help='How many times to repeat an experiment')


    metrics_parser = subparsers.add_parser('metric', help='Metric calculation based on the snapshots produced by a blocking aggression minimization simulation experiment')
    metrics_parser.add_argument('aggression_threshold', type=float, help="The threshold in [0,1] that determines when a user becomes aggressive. An aggression score larger than the threshold means that the user is aggressive. Eg. = 0.4")
    metrics_parser.add_argument('metric_type', type=str, choices=['similarity', 'aggression'], help="The type of metric to calculate, similarity metrics or aggression. 'aggression' is suggested for blocking aggression minimization experiment")

    args = parser.parse_args()
    print()
    print(args)

    if args.mode == 'simulation':
        experiment(args, False)
    elif args.mode == 'metric':
        calc_metrics(args, 'blocking')
