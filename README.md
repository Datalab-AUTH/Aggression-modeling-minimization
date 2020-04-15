# Aggression Modeling and Minimization Framework

A simulation framework of online aggression modeling and minimization
in online social networks and specifically Twitter. There are 3 distinct simulation
processes: 
   1. aggression diffusion modeling through IC and LT models.
   2. aggression minimization through competitive IC and LT models.
   3. aggression minimization through blocking/immunization methods.

This framework is the result of a master thesis and research work published in - CONFERENCE HERE -

More information on the parameters and values of the experiments can be found in - PAPER REFERENCE HERE - 

## Getting Started

This framework is written in Python 3.6

### Prerequisites

The used libraries and dependencies are included in requirements.txt located under the root folder.
After cloning the project run

```
pip install -r requirements.txt
```

to install the required dependencies.

## Execution

Each one of the above simulation processes run from a separate script. In particular:
   1. aggression_modeling.py runs the aggression modeling simulations
   2. agg_min_competitive.py runs the competitive aggression minimization simulations
   3. agg_min_blocking.py runs the blocking aggression minimization simulations

### Aggression modeling execution

The command to run aggression modeling simulation is:

```
python aggression_modeling.py simulation graph snapshot strategy ic|lt activation|threshold num_experiments
```

where:
   
   - simulation: executes a simulation experiment. This is the actual parameter value
   - graph: path to the graph file
   - snapshot: if True saves the intermediate steps of the process (required for metric calculation later)
   - strategy: seed node selection strategy. Options: ['r', 'aa', 'sd', 'dd']
   - ic|lt: whether to run IC or LT based aggression diffusion process
   - activation|threshold: activation criterion of IC - options: ['random', 'top', 'cumulative'] - or threshold strategy of LT - options: ['aggression', 'power']
   - num_experiments: how many times to run the process

Given that an aggression modeling simulation experiment has run, metrics can be produced with:

```
python aggression_modeling.py metric aggression_threshold metric_type
```

where:
   
   - metric: executes the metric calculation. This is the actual parameter value
   - aggression_threshold: threshold in [0,1] that determines when a user becomes aggressive.
   - metric_type: type of metric to calculate, similarity metrics or aggression - options: ['similarity', 'aggression']. 'similarity' is suggested for this type of experiment

### Competitive aggression minimization execution

In competitive aggression minimization there are 2 cascades, a negative (the one transferring aggression)
and a positive (the educational). The positive one aims at minimizing the effect of the negative cascade through a healing mechanism.

Currently, the negative cascade and the basic parameters of the positive cascade are determined according to
the best configuration found during the author's experimental process. If this needs to be changed, the code has to be altered.

The command to run competitive aggression minimization simulation is:

```
python agg_min_competitive.py simulation graph snapshot strategy ic|lt healing num_experiments
```

where:
   
   - simulation: executes a simulation experiment. This is the actual parameter value
   - graph: path to the graph file
   - snapshot: if True saves the intermediate steps of the process (required for metric calculation later)
   - strategy: positive cascade seed node selection strategy. Options: ['r', 'aa', 'sd', 'dd']
   - ic|lt: whether to run IC or LT based competitive cascades
   - healing: healing strategy for the positive cascade during the diffusion process. Options: ['vaccination', 'transfer', 'decay', 'hybrid']
   - num_experiments: how many times to run the process

Given that a competitive aggression minimization simulation experiment has run, metrics can be produced with:

```
python agg_min_competitive.py metric aggression_threshold metric_type
```

where:
   
   - metric: executes the metric calculation. This is the actual parameter value
   - aggression_threshold: the threshold in [0,1] that determines when a user becomes aggressive.
   - metric_type: the type of metric to calculate, similarity metrics or aggression - choices: ['similarity', 'aggression']. 'aggression' is suggested for this type of experiment

### Blocking aggression minimization execution

In blocking aggression minimization there is a single cascade, the aggression or negative.
The aim is to remove nodes or edges according to some strategy in order to reduce the overall aggression score at the end of the diffusion process.

The command to run blocking aggression minimization simulation is:

```
python agg_min_blocking.py simulation graph snapshot seedsize strategy adjacency seedstrategy ic|lt activation|threshold num_experiments
```

where:
   
   - simulation: executes a simulation experiment. This is the actual parameter value
   - graph: path to the graph file
   - snapshot: if True saves the intermediate steps of the process (required for metric calculation later)
   - seedsize: number of nodes or edges that the blocking mechanism will remove
   - strategy: whether to remove nodes or edges. Options: ['node', 'edge']. Node removal is more intrusive
   - adjacency: Type of adjacency. Options: ['normal', 'aggression']. Normal adjacency leads to abusive node or edge detection, while aggression adjacency is fine-tuned towards aggression detection
   - seedstrategy: seed node selection strategy for the cascade. Options: ['r', 'aa', 'sd', 'dd']
   - ic|lt: whether to run IC or LT based aggression diffusion process
   - activation|threshold: activation criterion of IC - options: ['random', 'top', 'cumulative'] - or threshold strategy of LT - options: ['aggression', 'power']
   - num_experiments: how many times to run the process

Given that a blocking aggression minimization simulation experiment has run, metrics can be produced with:

```
python agg_min_blocking.py metric aggression_threshold metric_type
```

where:
   
   - metric: executes the metric calculation. This is the actual parameter value
   - aggression_threshold: threshold in [0,1] that determines when a user becomes aggressive.
   - metric_type: type of metric to calculate, similarity metrics or aggression - options: ['similarity', 'aggression']. 'aggression' is suggested for this type of experiment

### Graph generation

The graphs used in these experiment were produced using the code under graph_creation
directory

### Plots

plot_factory.py is a file containing various plot used by the authors. These are not generic plots but 
rather fine-tuned plots towards specific needs. You can use them as an idea or build upon them.
## Authors

* **Ph.D. candidate Marinos Poiitis, AUTH** - *Main work* - [mpoiitis](https://github.com/mpoiitis)
* **Dr. Nicolas Kourtellis, Telefonica I+D, Spain** - *Supervisor*
* **Pr. Athena Vakali, AUTH** - *Supervisor*

\* AUTH: Aristotle University of Thessaloniki, Greece

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details