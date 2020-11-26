import argparse
import time
import networkx as nx
import seed_selection
import utility
from metrics import calc_metrics
from models.influence_minimization.competitive_ic import run_ic
from models.influence_minimization.competitive_lt import run_lt


def run_experiment(g: nx.DiGraph, args, instance: int):
    """
    The actual experiment
    Values for negative cascade are hard coded. They represent the best experiment of aggression modeling phase
    :param instance: current repeat of the experiment
    :param args: command-line arguments
    :param g: the graph
    """

    # Attach aggression related attributes to nodes
    g = utility.insert_aggression(g)

    # negative seeds are selected according to the best strategy of aggression modeling phase
    negative_seeds = seed_selection.single_discount(g, args.seedsize)

    # Determine the seed selection strategy for positive seeds
    if args.strategy == 'aa':
        positive_seeds = seed_selection.all_aggressive(g)
    elif args.strategy == 'ta':
        positive_seeds = seed_selection.top_aggressive(g, args.seedsize)
    elif args.strategy == 'sd':
        positive_seeds = seed_selection.single_discount(g, args.seedsize)
    elif args.strategy == 'dd':
        positive_seeds = seed_selection.degree_discount(g, args.seedsize)
    else:
        positive_seeds = seed_selection.random(g, args.seedsize)

    if args.model == 'ic':
        n_activated, p_activated, agg_scores = run_ic(g, negative_seeds, positive_seeds, args, instance)
    else:  # lt
        n_activated, p_activated, agg_scores = run_lt(g, negative_seeds, positive_seeds, args, instance)

    return n_activated, p_activated, agg_scores


def experiment(args, save):
    """
    The wrapper experiment function
    """

    graph = args.graph  # 'data/graphs/Twitter_power'

    print('Experiment will be repeated {} times.'.format(args.num_experiments))
    print()

    # Network topology
    g = nx.read_gpickle(graph)

    # actual experiment
    for i in range(args.num_experiments):
        start = time.time()

        n_activated, p_activated, agg_scores = run_experiment(g, args, i)

        total_time = time.time() - start
        print("Total time: ", total_time)
        print()

        if save:
            utility.write_to_csv_min_competitive(args, n_activated, p_activated, agg_scores[0], agg_scores[-1], total_time)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Competitive Aggression Minimization simulation framework')
    subparsers = parser.add_subparsers(dest='mode', help="simulation or metric. Simulation runs the actual simulations"
                                                         " of the diffusion experiment and produces the intermediate snapshots."
                                                         " Metric uses the snapshots to calculate cosine similarity, Pearson R, Spearman R, Kendal Tau and aggression"
                                                         " needed for the various plots")

    simulation_parser = subparsers.add_parser('simulation', help="Simulation of a competitive aggression minimization process based on the given parameters. "
                                                   "The negative cascade and the basic parameters of the positive cascade are determined according to "
                                                   " the best configuration found during the author's experimental process. If this needs to be changed, the code has to be altered")
    simulation_parser.add_argument("graph", type=str, help="Path to graph pickle file")
    simulation_parser.add_argument("snapshot", type=bool, help='If True, apart from the first and last step, it will create intermediate snapshots during the process')
    simulation_parser.add_argument("seedsize", type=int, help="Seed set size. It is the number of initial infected nodes. Same for both positive and negative cascade. 5594 is the total number of aggressive users")
    simulation_parser.add_argument("strategy", type=str, choices=['r', 'aa', 'sd', 'dd'], help="Seed node selection strategy. Short names for 'Random', 'All aggressive', 'Single Discount' and 'Degree Discount'")

    simulation_parser.add_argument("model", type=str, choices=['ic', 'lt'], help="The diffusion model both for positive and negative cascade")
    simulation_parser.add_argument('healing', type=str, choices=['vaccination', 'transfer', 'decay', 'hybrid'], help='Healing strategy for the positive cascade during the diffusion process')
    simulation_parser.add_argument('num_experiments', type=int, help='How many times to repeat an experiment')

    metrics_parser = subparsers.add_parser('metric', help='Metric calculation based on the snapshots produced by a competitive aggression minimization simulation experiment')
    metrics_parser.add_argument('aggression_threshold', type=float, help="The threshold in [0,1] that determines when a user becomes aggressive. An aggression score larger than the threshold means that the user is aggressive. Eg. = 0.4")
    metrics_parser.add_argument('metric_type', type=str, choices=['similarity', 'aggression'], help="The type of metric to calculate, similarity metrics or aggression. 'aggression' is suggested for competitive aggression minimization experiment")

    args = parser.parse_args()
    print()
    print(args)

    if args.mode == 'simulation':
        experiment(args, True)
    elif args.mode == 'metric':
        calc_metrics(args, 'minimization')
