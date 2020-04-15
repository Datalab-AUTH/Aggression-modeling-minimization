import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import statsmodels.api as sm
import matplotlib.ticker as mtick
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison
import seaborn as sns
import numpy as np
import utility
import os
import random
from heatmap import corrplot

def min_blocking_evolution_plot(file='results/min_blocking_evolution.csv', save=False):
    names = [str(i) for i in range(49)]  # dummy column names to avoid problem with csv reading
    data = pd.read_csv(file, header=None, names=names)

    # Initialize the figure
    plt.style.use('seaborn-darkgrid')
    # create a color palette
    palette = plt.get_cmap('Set1')

    num = 0
    for idx, row in data.iterrows():
        line_data = row.iloc[1:]
        label = row.iloc[0]
        plt.plot(line_data, color=palette(num), linewidth=1.9, alpha=0.9, label=label)
        num += 1

    plt.legend()
    plt.xlabel('Steps')
    plt.ylabel('Aggression Score')

    plt.tight_layout()
    if save:
        plt.savefig('results/plots/evolutions/minimization/{}.png'.format(file[7:-4]), format='png',
                    bbox_inches='tight')

    plt.show()

### NEW PLOTS ###

def get_dir_list(param, exp_type, model='ic'):
    dirs = os.listdir('snapshots/{}'.format(exp_type))
    dirs = [directory for directory in dirs if model in directory]

    dir_list = list()  # contains tuples of experiments that differ only in one parameter in order to create cdf
    if param == 'graph':
        power_experiments = [directory for directory in dirs if '_P_' in directory]
        weighted_experiments = [directory for directory in dirs if '_W_' in directory]
        jaccard_experiments = [directory for directory in dirs if '_J_' in directory]
        for i in range(len(power_experiments)):
            dir_list.append((power_experiments[i], weighted_experiments[i], jaccard_experiments[i]))
    elif param == 'seed strategy':
        aa_experiments = [directory for directory in dirs if '_aa_' in directory]
        dd_experiments = [directory for directory in dirs if '_dd_' in directory]
        r_experiments = [directory for directory in dirs if '_r_' in directory]
        sd_experiments = [directory for directory in dirs if '_sd_' in directory]
        for i in range(len(aa_experiments)):
            if model == 'ic':
                dir_list.append((aa_experiments[i], r_experiments[i], sd_experiments[i], dd_experiments[i]))
            else:
                dir_list.append((aa_experiments[i], r_experiments[i], sd_experiments[i]))
    elif param == 'activation':
        c_experiments = [directory for directory in dirs if directory.endswith('_c')]
        r_experiments = [directory for directory in dirs if directory.endswith('_r')]
        t_experiments = [directory for directory in dirs if directory.endswith('_t')]
        for i in range(len(c_experiments)):
            dir_list.append((c_experiments[i], r_experiments[i], t_experiments[i]))
    elif param == 'threshold':
        p_experiments = [directory for directory in dirs if directory.endswith('_p')]
        a_experiments = [directory for directory in dirs if directory.endswith('_a')]
        for i in range(len(p_experiments)):
            dir_list.append((p_experiments[i], a_experiments[i]))
    elif param == 'healing':
        d_experiments = [directory for directory in dirs if directory.endswith('_d')]
        h_experiments = [directory for directory in dirs if directory.endswith('_h')]
        t_experiments = [directory for directory in dirs if directory.endswith('_t')]
        v_experiments = [directory for directory in dirs if directory.endswith('_v')]
        for i in range(len(d_experiments)):
            dir_list.append((d_experiments[i], h_experiments[i], t_experiments[i], v_experiments[i]))
    elif param == 'blocking':
        e_a_experiments = [directory for directory in dirs if directory.startswith(model) and directory.endswith('_aggression') and 'edge' in directory]
        e_n_experiments = [directory for directory in dirs if directory.startswith(model) and directory.endswith('_normal') and 'edge' in directory]
        n_a_experiments = [directory for directory in dirs if directory.startswith(model) and directory.endswith('_aggression') and 'node' in directory]
        n_n_experiments = [directory for directory in dirs if directory.startswith(model) and directory.endswith('_normal') and 'node' in directory]
        for i in range(len(e_a_experiments)):
            dir_list.append((e_a_experiments[i], e_n_experiments[i], n_a_experiments[i], n_n_experiments[i]))
    else:
        return

    return dir_list


def plot_evolution(configuration, threshold=None, exp_type='modeling', metric_type='similarity', save=False):
    """
    :param metric_type: type of metric. similarity or aggression
    :param exp_type: type of experiment
    :param configuration: for which configuration to plot snapshot evolution.
    :param threshold: for which threshold to plot metrics. Options: [0.05, 0.1, 0.2, 0.5, 0.9]
    :param save: whether to save the plot or not
    """
    colors = ['#e66101', '#fdb863', '#5e3c99', '#b2abd2']
    dirs = os.listdir('snapshots/{}/{}'.format(exp_type, configuration))

    metric_dict = dict()
    for directory in dirs:
        if metric_type == 'similarity':
            filename = 'snapshots/{}/{}/{}/total_{}.csv'.format(exp_type, configuration, directory, threshold)
        elif metric_type == 'aggression':
            filename = 'snapshots/{}/{}/{}/total_aggressions.csv'.format(exp_type, configuration, directory)
        else:
            return

        df = pd.read_csv(filename, header=None)

        for index, row in df.iterrows():
            if metric_type == 'similarity':
                if index == 0:
                    metric = 'Cosine Similarity'
                elif index == 1:
                    metric = 'Pearson R'
                else:
                    metric = 'Spearman R'
            else:
                metric = 'Aggression Score'

            # first column contains metric's name, so discard it
            values = row.iloc[1:]
            if metric in metric_dict.keys():
                sum_values, col_frequencies = metric_dict[metric]

                for idx, val in enumerate(values):
                    if len(sum_values) > idx:
                        sum_values[idx] += val
                        col_frequencies[idx] += 1
                    else:
                        sum_values.append(val)
                        col_frequencies.append(1)
                metric_dict.update({metric: (sum_values, col_frequencies)})
            else:
                sum_values = list()
                col_frequencies = list()
                for val in values:
                    sum_values.append(val)
                    col_frequencies.append(1)
                metric_dict.update({metric: (sum_values, col_frequencies)})

    metric_means = dict()
    for metric in metric_dict.keys():
        mean_values = [val / freq for val, freq in zip(*metric_dict[metric])]
        metric_means.update({metric: mean_values})

    fig = plt.figure()
    ax = plt.subplot(111)

    for idx, metric in enumerate(metric_means.keys()):
        ticks = np.arange(0, len(metric_means[metric]))
        ax.plot(ticks, metric_means[metric], color=colors[idx], label=metric)

    plt.xlabel('Steps')
    plt.ylabel('Magnitude')
    if threshold:
        ax.set_title('Configuration:{} Threshold:{}'.format(configuration, threshold))
    else:
        ax.set_title('Configuration:{}'.format(configuration))
    ax.legend(loc='best')  # put legend outside of plot

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # force integer x ticks
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.tight_layout()

    if save:
        directory = 'results/plots/snapshots/{}/'.format(exp_type)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if metric_type == 'similarity':
            plt.savefig('{}/{}_{}.png'.format(directory, configuration, threshold), format='png', bbox_inches='tight')
        else:  # aggression
            plt.savefig('{}/{}_aggressions.png'.format(directory, configuration), format='png', bbox_inches='tight')
    plt.show()


def plot_evolution_best(save=False):
    """
    :param save: whether to save the plot or not
    """
    colors = ['#e66101', '#fdb863', '#5e3c99', '#b2abd2']
    dirs = os.listdir('snapshots/modeling/ic_J_sd_c')
    dirs2 = os.listdir('snapshots/modeling/lt_J_sd_a')

    metric_dict = dict()
    metric = 'Cosine Similarity'
    for directory in dirs:
        filename = 'snapshots/modeling/ic_J_sd_c/{}/total_0.4.csv'.format(directory)

        df = pd.read_csv(filename, header=None)

        cosines = df.iloc[0].iloc[1:]

        if metric in metric_dict.keys():
            sum_values, col_frequencies = metric_dict[metric]

            for idx, val in enumerate(cosines):
                if len(sum_values) > idx:
                    sum_values[idx] += val
                    col_frequencies[idx] += 1
                else:
                    sum_values.append(val)
                    col_frequencies.append(1)
            metric_dict.update({metric: (sum_values, col_frequencies)})
        else:
            sum_values = list()
            col_frequencies = list()
            for val in cosines:
                sum_values.append(val)
                col_frequencies.append(1)
            metric_dict.update({metric: (sum_values, col_frequencies)})

    metric_dict2 = dict()
    for directory in dirs2:
        filename = 'snapshots/modeling/ic_J_sd_c/{}/total_0.4.csv'.format(directory)

        df = pd.read_csv(filename, header=None)
        cosines = df.iloc[0].iloc[1:]

        if metric in metric_dict2.keys():
            sum_values, col_frequencies = metric_dict2[metric]

            for idx, val in enumerate(cosines):
                if len(sum_values) > idx:
                    sum_values[idx] += val
                    col_frequencies[idx] += 1
                else:
                    sum_values.append(val)
                    col_frequencies.append(1)
            metric_dict2.update({metric: (sum_values, col_frequencies)})
        else:
            sum_values = list()
            col_frequencies = list()
            for val in cosines:
                sum_values.append(val)
                col_frequencies.append(1)
            metric_dict2.update({metric: (sum_values, col_frequencies)})

    metric_means = dict()
    for metric in metric_dict.keys():
        mean_values = [val / freq for val, freq in zip(*metric_dict[metric])]
        metric_means.update({metric: mean_values})

    metric_means2 = dict()
    for metric in metric_dict2.keys():
        mean_values = [val / freq for val, freq in zip(*metric_dict2[metric])]
        metric_means2.update({metric: mean_values})

    means1 = metric_means[metric]
    means2 = metric_means2[metric]

    fig, ax = plt.subplots()

    ticks = np.arange(0, len(means1))
    ax.plot(ticks, means1, color=colors[0], label='ic_J_sd_c')

    ticks = np.arange(0, len(means2))
    ax.plot(ticks, means2, color=colors[1], label='lt_J_sd_a')

    plt.xlabel('Steps')
    plt.ylabel('Magnitude')
    plt.legend(loc='best')

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # force integer x ticks
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.tight_layout()

    if save:
        directory = 'results/plots/snapshots/modeling/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig('{}/best_0_4.png'.format(directory), format='png', bbox_inches='tight')
    plt.show()


def min_evolution_plot(model, exp_type='minimization', save=False):

    if exp_type == 'minimization':
        param = 'healing'
    else:
        param = 'blocking'

    # create a color palette
    colors = ['#e66101', '#fdb863', '#b2abd2', '#5e3c99']
    linestyles = ['-.', '--', '-', ':']

    if exp_type == 'blocking':
        fig, axes = plt.subplots(nrows=1, ncols=1)
    else:
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)

    dirs = get_dir_list(param, exp_type, model)
    dirs = [dir_tuple for dir_tuple in dirs if model in dir_tuple[0]]
    for idx, dir_tuple in enumerate(dirs):
        if exp_type == 'minimization':
            seed_strategy = dir_tuple[0].split('_')[2]  # eg 'aa'

        if model == 'ic':
            directory = 'snapshots/modeling/ic_J_sd_c/0'
        else:
            directory = 'snapshots/modeling/lt_J_sd_a/0'
        steps = os.listdir(directory)
        steps = [step for step in steps if 'total' not in step]

        no_heal_scores = list()
        for step in steps:
            no_heal_data_step = pd.read_csv('{}/{}'.format(directory, step), names=['User', 'Aggression Score'])
            aggression_score = no_heal_data_step['Aggression Score'].sum()
            no_heal_scores.append(aggression_score)
        no_heal_scores_percent = [(x - no_heal_scores[idx]) / no_heal_scores[idx] * 100 for idx, x in enumerate(no_heal_scores)]
        if idx == 0:
            if exp_type == 'blocking':
                axes.plot(no_heal_scores_percent, color='black', alpha=0.9, label='No immunization')
            else:
                axes[idx % 2, idx // 2].plot(no_heal_scores_percent, color='black', alpha=0.9, label='No healing')
        else:
            if exp_type == 'blocking':
                axes.plot(no_heal_scores_percent, color='black', alpha=0.9)
            else:
                axes[idx % 2, idx // 2].plot(no_heal_scores_percent, color='black', alpha=0.9)

        if exp_type == 'blocking':
            axes.set_xticks(np.arange(0, len(no_heal_scores), 3))
            axes.yaxis.set_major_formatter(mtick.PercentFormatter())
        else:
            axes[idx % 2, idx // 2].set_xticks(np.arange(0, len(no_heal_scores), 3))
            axes[idx % 2, idx // 2].yaxis.set_major_formatter(mtick.PercentFormatter())

        for index, directory in enumerate(dir_tuple):
            if param == 'healing':
                legend_label = directory.split('_')[-1]
            else:
                splits = directory.split('_')
                legend_label = '_'.join([splits[2], splits[-1]])
            repeats = os.listdir('snapshots/{}/{}'.format(exp_type, directory))
            for repeat in repeats:
                filename = 'snapshots/{}/{}/{}/total_aggressions.csv'.format(exp_type, directory, repeat)
                df = pd.read_csv(filename, header=None)
                scores = df.iloc[0, 1:].tolist()
                scores = [(x - no_heal_scores[idx])/no_heal_scores[idx] * 100 if idx < len(no_heal_scores) else (x - no_heal_scores[-1])/no_heal_scores[-1] * 100
                          for idx, x in enumerate(scores)]
                if idx == 0:
                    if exp_type == 'blocking':
                        axes.plot(scores, color=colors[index], linestyle=linestyles[index], linewidth=1.9, alpha=0.9,
                                  label=legend_label)
                    else:
                        axes[idx % 2, idx // 2].plot(scores, color=colors[index], linestyle=linestyles[index], linewidth=1.9, alpha=0.9,
                                                 label=legend_label)
                else:
                    if exp_type == 'blocking':
                        axes.plot(scores, color=colors[index], linestyle=linestyles[index], linewidth=1.9, alpha=0.9)
                    else:
                        axes[idx % 2, idx // 2].plot(scores, color=colors[index], linestyle=linestyles[index], linewidth=1.9, alpha=0.9)

                if exp_type == 'minimization':
                    axes[idx % 2, idx // 2].set_title(seed_strategy, loc='left', fontsize=12, fontweight=0)

                if exp_type == 'blocking':
                    axes.set_xticks(np.arange(0, len(scores), 3))
                    axes.yaxis.set_major_formatter(mtick.PercentFormatter())
                else:
                    axes[idx % 2, idx // 2].set_xticks(np.arange(0, len(scores), 3))
                    axes[idx % 2, idx // 2].yaxis.set_major_formatter(mtick.PercentFormatter())


    fig.text(0.5, 0.01, 'Steps', ha='center')
    fig.text(0.01, 0.5, 'Aggression Score', va='center', rotation='vertical')
    if exp_type == 'minimization':
        title = 'Healing Strategies'
        axes[0, 0].legend(title=title, loc='lower left', bbox_to_anchor=(0.0, 1.1), ncol=3, frameon=False)
    else:
        title = 'Strategies'
        axes.legend(title=title, loc='lower left', bbox_to_anchor=(0.0, 1.1), ncol=3, frameon=False)

    fig.tight_layout()
    if save:
        directory = 'results/plots/evolutions/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig('{}/{}_{}.png'.format(directory, model, exp_type), format='png', bbox_inches='tight')

    plt.show()


def plot_activations(model='ic', exp_type='modeling', plot_type='scatter', save=False):
    if exp_type == 'modeling':
        data = pd.read_csv('results/modeling/total.csv')
        # contains unnecessary weighting schemes so remove them
        data = data[data['Graph'] == 'data/graphs/Twitter_jaccard']
        data = data[data['Model'] == model]
        activated = data['Activated Nodes']
    elif exp_type == 'minimization':
        data = pd.read_csv('results/competitive/total.csv')
        data = data[data['Model'] == model]
        activated = data['Negative Activated Nodes']
    else:
        data = pd.read_csv('results/blocking/total.csv')
        data = data[data['Diffusion Model'] == model]
        activated = data['Activated Nodes']

    configurations = list()
    for idx, row in data.iterrows():
        if exp_type == 'modeling':
            conf = utility.create_config_string(row=row)
        elif exp_type == 'minimization':
            conf = utility.create_config_string_min_competitive(row=row)
        else:
            conf = utility.create_config_string(row=row, blocking=True)
        configurations.append(conf)
        configurations = [conf for conf in configurations if model in conf]

    if plot_type == 'scatter':
        plt.scatter(configurations, activated)
    else:
        plt.bar(configurations, activated, color='#5e3c99')
    if exp_type == 'modeling' or exp_type == 'blocking':
        plt.ylabel('Activated Nodes')
    else:
        plt.ylabel('Negative Activated Nodes')

    if exp_type == 'blocking':
        plt.xticks(rotation=45)
    else:
        plt.xticks(rotation=90)

    if save:
        directory = 'results/plots/activations/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig('{}/{}_{}.png'.format(directory, model, exp_type), format='png', bbox_inches='tight')

    plt.show()


def metric_cdf(param, threshold, metric, exp_type='modeling', save=False):
    """
    Plot the cdf for each configuration according to the convergence value of each experiment
    :param exp_type: type of experiment
    :param param: which parameter to plot. Options: ['graph', 'seed strategy', 'activation']
    :param threshold: for which threshold to plot metrics. Options: [0.05, 0.1, 0.2, 0.5, 0.9]
    :param metric: which metric to plot. Options: ['cosine', 'pearson', 'spearman']
    :param save: whether to save the plot or not
    """

    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 18

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    dir_list = get_dir_list(param, exp_type)
    if not dir_list:
        return

    for t in dir_list:  # a cdf for each tuple

        exp_metric_dict = dict()  # dictionary with (experiment: list of max 'metric' values) items
        for experiment in t:
            repeats = os.listdir('snapshots/{}/{}'.format(exp_type, experiment))
            convergence_values = list()
            for repeat in repeats:
                filename = 'snapshots/{}/{}/{}/total_{}.csv'.format(exp_type, experiment, repeat, threshold)
                df = pd.read_csv(filename, header=None)

                # metrics are located in the first column. keep only the row of the specified metric
                row_idx = df.index[df.iloc[:, 0] == metric][0]
                values = df.iloc[row_idx, 1:].tolist()
                convergence_value = values[-1]
                convergence_values.append(convergence_value)
            exp_metric_dict.update({experiment: convergence_values})

        colors = ['#e66101', '#fdb863', '#b2abd2', '#5e3c99']
        idx = 0

        for experiment, values in exp_metric_dict.items():
            values.sort()  # sort values
            s = float(sum(values))
            cdf = np.cumsum(values) / s  # calculate cdf
            plt.plot(values, cdf, label=experiment, color=colors[idx])
            idx += 1
        plt.legend(loc='best')
        plt.title('Threshold={}'.format(threshold))
        plt.ylabel('CDF')
        if metric == 'cosine':
            plt.xlabel('Cosine Similarity')
        elif metric == 'pearson':
            plt.xlabel('Pearson R')
        elif metric == 'spearman':
            plt.xlabel('Spearman R')
        else:
            return
        plt.tight_layout()
        if save:
            directory = 'results/plots/cdfs/{}'.format(param)
            if not os.path.exists(directory):
                os.makedirs(directory)

            if param == 'graph':
                string_to_write = t[0][5:]
            elif param == 'seed strategy':
                string_to_write_1 = t[0][3:4]
                string_to_write_2 = t[0][-1]
                string_to_write = '_'.join([string_to_write_1, string_to_write_2])
            elif param == 'activation':
                string_to_write = t[0][3:7]
            else:
                return
            plt.xticks(rotation=30)
            plt.savefig('{}/{}_{}_{}.png'.format(directory, string_to_write, metric, threshold), format='png', bbox_inches='tight')
        plt.show()


def significance_plot(param, threshold, metric, exp_type='modeling', save=False):
    """
    :param exp_type: type of experiment
    :param param: which parameter to plot. Options: ['graph', 'seed strategy', 'activation']
    :param threshold: for which threshold to plot metrics. Options: [0.05, 0.1, 0.2, 0.5, 0.9]
    :param metric: which metric to plot. Options: ['cosine', 'pearson', 'spearman']
    :param save: whether to save the plot or not
    """

    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 18

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    dir_list = get_dir_list(param,exp_type)
    if not dir_list:
        return

    for t in dir_list:  # a cdf for each tuple

        exp_metric_dict = dict()  # dictionary with (experiment: list of max 'metric' values) items
        for experiment in t:
            repeats = os.listdir('snapshots/{}/{}'.format(exp_type, experiment))
            convergence_values = list()
            for repeat in repeats:
                filename = 'snapshots/{}/{}/{}/total_{}.csv'.format(exp_type, experiment, repeat, threshold)
                df = pd.read_csv(filename, header=None)

                # metrics are located in the first column. keep only the row of the specified metric
                row_idx = df.index[df.iloc[:, 0] == metric][0]
                values = df.iloc[row_idx, 1:].tolist()
                convergence_value = values[-1]
                convergence_values.append(convergence_value)
            exp_metric_dict.update({experiment: convergence_values})

        # create dataframe needed for ols model
        values = list()
        config = list()
        for experiment, vals in exp_metric_dict.items():
            for value in vals:
                values.append(value)
                config.append(experiment)
        data = {'experiment': config, 'value': values}
        df = pd.DataFrame(data)

        # calculate an overall statitistics table and watch f-statistic and p-value to see if there is significance
        results = ols('value ~ C(experiment)', data=df).fit()
        print(results.summary())
        # aov_table = sm.stats.anova_lm(results, typ=2)
        # print(utility.anova_table(aov_table))

        plt.style.use('seaborn-darkgrid')
        # To test between groups, we need to do some post-hoc testing
        mc = MultiComparison(df['value'], df['experiment'])
        mc_results = mc.tukeyhsd()
        # print(mc_results)
        mc_results.plot_simultaneous()

        plt.tight_layout()
        plt.ylabel('Configuration')
        plt.xlabel('Magnitude')

        if save:
            directory = 'results/plots/statistics/{}'.format(param)
            if not os.path.exists(directory):
                os.makedirs(directory)

            if param == 'graph':
                string_to_write = t[0][5:]
            elif param == 'seed strategy':
                string_to_write_1 = t[0][3:4]
                string_to_write_2 = t[0][-1]
                string_to_write = '_'.join([string_to_write_1, string_to_write_2])
            elif param == 'activation':
                string_to_write = t[0][3:7]
            else:
                return
            plt.savefig('{}/{}_{}_{}_tukey.png'.format(directory, string_to_write, metric, threshold), format='png', bbox_inches='tight')
        plt.show()


def lt_metrics(threshold, metric, exp_type='modeling', save=False):
    """
        :param exp_type: experiment type
        :param remove_outliers: whether or not to remove outlier values from plot
        :param threshold: for which threshold to plot metrics. Options: [0.05, 0.1, 0.2, 0.5, 0.9]
        :param metric: which metric to plot. Options: ['cosine', 'pearson', 'spearman']
        :param save: whether to save the plot or not
        """
    dirs = os.listdir('snapshots/{}'.format(exp_type))
    dirs = [directory for directory in dirs if 'lt' in directory]

    conf_metric_dict = dict()
    for directory in dirs:  # a cdf for each tuple
        repeats = os.listdir('snapshots/{}/{}'.format(exp_type, directory))

        mean_values = list()
        col_frequencies = list()  # counts how many times a column index appears. needed to calculate mean
        for repeat in repeats:
            filename = 'snapshots/{}/{}/{}/total_{}.csv'.format(exp_type, directory, repeat, threshold)
            df = pd.read_csv(filename, header=None)

            # metrics are located in the first column. keep only the row of the specified metric
            row_idx = df.index[df.iloc[:, 0] == metric][0]
            values = df.iloc[row_idx, 1:]
            for idx, val in enumerate(values):
                if len(mean_values) > idx:
                    mean_values[idx] += val
                    col_frequencies[idx] += 1
                else:
                    mean_values.append(val)
                    col_frequencies.append(1)

        mean_values = [val / freq for val, freq in zip(mean_values, col_frequencies)]
        convergence_value = mean_values[-1]
        conf_metric_dict.update({directory: convergence_value})

    plt.style.use('seaborn-darkgrid')
    plot = plt.scatter(conf_metric_dict.keys(), conf_metric_dict.values())
    plt.xlabel('Configuration')
    plt.xticks(rotation=90)
    plt.ylim(0.6850, 0.6950)
    if metric == 'cosine':
        plt.ylabel('Cosine Similarity')
    elif metric == 'pearson':
        plt.ylabel('Pearson R')
    elif metric == 'spearman':
        plt.ylabel('Spearman R')
    else:
        return

    plot.set_label("$T_A$" + " = {}".format(threshold))
    plt.legend()
    plt.tight_layout()

    if save:
        directory = 'results/plots/lt'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig('{}/{}_comparison_{}.png'.format(directory, metric, threshold), format='png',
                    bbox_inches='tight')
    plt.show()


# configurations = os.listdir('snapshots/modeling')
# for configuration in configurations:
#     plot_evolution(configuration, threshold=0.4, exp_type='modeling', metric_type='similarity', save=True)

plot_evolution_best(save=True)

# configurations = os.listdir('snapshots/minimization')
# for configuration in configurations:
#     plot_evolution(configuration, threshold=None, exp_type='minimization', metric_type='aggression', save=True)


# for threshold in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.1]:
#     for metric in ['cosine', 'pearson', 'spearman']:
#         lt_metrics(threshold, metric, 'modeling', False)


# for param in ['graph', 'seed strategy', 'activation']:
#     for threshold in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.1]:
#         for metric in ['cosine', 'pearson', 'spearman']:
#             metric_cdf(param, threshold, metric, exp_type='modeling', save=True)

# for param in ['graph', 'seed strategy', 'activation']:
#     for threshold in [0.4]:
#         for metric in ['cosine', 'pearson', 'spearman']:
#             significance_plot(param, threshold, metric, exp_type='modeling', save=False)


# for m in ['ic', 'lt']:
#     for t in ['minimization', 'blocking']:
#         plot_activations(model=m, exp_type=t, plot_type='bar', save=True)
#         min_evolution_plot(model=m, exp_type=t, save=True)
