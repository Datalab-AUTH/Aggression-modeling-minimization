import argparse
import time
import networkx as nx
from numpy import mean
import seed_selection
import utility
from metrics import calc_metrics

from models.influence_maximization.ic import run_ic
from models.influence_maximization.lt import run_lt


def run_experiment(g: nx.DiGraph, args, instance: int):
    """
    The main function that runs an experiment according to the given parameters
    :param instance: current repeat of the experiment
    :param args: command-line arguments
    :param g: the graph
    """

    if args.model == 'ic':
        assert args.activation, \
            "Activation strategy should be defined. Options: ['random', 'top', 'cumulative']"
    if args.model == 'lt':
        assert args.threshold, \
            "Threshold strategy should be defined. Options: ['aggression', 'power', 'random']"

    # Attach aggression related attributes to nodes
    g = utility.insert_aggression(g)

    # Determine the seed selection strategy
    # for strategies that need seed size we give 5594 as it was found via aggression threshold to return 8% of users.
    if args.strategy == 'aa':
        seeds = seed_selection.all_aggressive(g)
    elif args.strategy == 'ta':
        seeds = seed_selection.top_aggressive(g, args.seedsize)
    elif args.strategy == 'sd':
        seeds = seed_selection.single_discount(g, args.seedsize)
    elif args.strategy == 'dd':
        seeds = seed_selection.degree_discount(g, args.seedsize)
    else:
        seeds = seed_selection.random(g, args.seedsize)

    if args.model == 'ic':
        activated, agg_scores = run_ic(g, seeds, args, instance)
    else:  # lt
        activated, agg_scores = run_lt(g, seeds, args, instance)

    return activated,  agg_scores


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
            utility.write_to_csv_modeling(args, activated, agg_scores[0], agg_scores[-1], total_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggression Diffusion simulation framework')

    subparsers = parser.add_subparsers(dest='mode', help="simulation or metric. Simulation runs the actual simulations"
                                            " of the diffusion experiment and produces the intermediate snapshots."
                                            " Metric uses the snapshots to calculate cosine similarity, Pearson R, Spearman R, Kendal Tau and aggression"
                                            " needed for the various plots")

    simulation_parser = subparsers.add_parser('simulation', help='Simulation of an aggression diffusion process based on the given parameters')
    simulation_parser.add_argument("graph", choices=['data/graphs/Twitter_jaccard', 'data/graphs/Twitter_power', 'data/graphs/Twitter_weighted_overlap', 'data/graphs/Twitter_random'], type=str, help="Path to graph pickle file")
    simulation_parser.add_argument("seedsize", type=int, help="Seed set size. It is the number of initial infected nodes. 5594 is the total number of aggressive users")
    simulation_parser.add_argument("strategy", type=str, choices=['r', 'aa', 'ta', 'sd', 'dd'], help="Seed node selection strategy. Short names for 'Random', 'All aggressive', 'Single Discount' and 'Degree Discount'")
    simulation_parser.add_argument("--snapshot", dest='snapshot', action='store_true', help='If True, apart from the first and last step, it will create intermediate snapshots during the process')

    sim_subparsers = simulation_parser.add_subparsers(dest='model')

    ic_parser = sim_subparsers.add_parser('ic', help='Simulation of an aggression diffusion process based on IC model')
    ic_parser.add_argument('--activation', default='cumulative', type=str, choices=['random', 'top', 'cumulative'],
                        help="Activation strategy during diffusion process")
    ic_parser.add_argument('--num_experiments', default=10, type=int, help='How many times to repeat an experiment. > 1 is suggested'
                                                              ' as IC is probabilistic')

    lt_parser = sim_subparsers.add_parser('lt', help='Simulation of an aggression diffusion process based on LT model')
    lt_parser.add_argument('--threshold', default='aggression', type=str, choices=['aggression', 'power', 'random'],
                        help="Node threshold strategy")
    lt_parser.add_argument('--num_experiments', default=1, type=int, help='How many times to repeat an experiment')


    metrics_parser = subparsers.add_parser('metric', help='Metric calculation based on the snapshots produced by an'
                                                          ' aggression diffusion simulation experiment')
    metrics_parser.add_argument('--aggression_threshold', default=0.4, type=float,
                                help="The threshold in [0,1] that determines when a user becomes aggressive."
                                " An aggression score larger than the threshold means that the user is aggressive. Eg. = 0.4")
    metrics_parser.add_argument('--metric_type', default='similarity', type=str, choices=['similarity', 'aggression'],
                                help="The type of metric to calculate, similarity metrics or aggression. "
                                     "'similarity' is suggested for aggression modeling experiment")
    metrics_parser.add_argument("--seedsize", default=None, type=int, help="Seed set size. It is the number of initial infected nodes. 5594 is the total number of aggressive users")
    metrics_parser.add_argument("--configuration", default=None, type=str, help="Specifies a single experiment to run metrics for")
    args = parser.parse_args()
    print()
    print(args)

    if args.mode == 'simulation':
        experiment(args, True)
    elif args.mode == 'metric':
        calc_metrics(args, 'modeling')

