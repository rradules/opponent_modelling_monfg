import matplotlib
import pandas as pd

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import mkdir_p

sns.set()
sns.despine()
sns.set_context("paper", rc={"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15,
                             "legend.fontsize": 16})
sns.set_style('white', {'axes.edgecolor': "0.5", "pdf.fonttype": 42})
plt.gcf().subplots_adjust(bottom=0.15, left=0.14)


def plot_results(game, mooc, path_data, experiment):
    path_plots = f'plots/tour_{experiment}_{game}_l{l1}_{l2}'
    mkdir_p(path_plots)

    df1 = pd.read_csv(f'{path_data}/agent1_payoff_{info}.csv')

    ax = sns.lineplot(x='Episode', y='Payoff', linewidth=2.0, data=df1, ci='sd',
                      label=f'Agent 1')
    df2 = pd.read_csv(f'{path_data}/agent2_payoff_{info}.csv')

    ax = sns.lineplot(x='Episode', y='Payoff', linewidth=2.0, data=df2, ci='sd',
                      label=f'Agent2')

    ax.set(ylabel='Scalarised payoff per step')
    ax.set(xlabel='Iterations')
    # ax.set_ylim(0, 14)
    ax.set_xlim(0, episodes)
    plot_name = f"{path_plots}/payoffs"
    # plt.title("Agent 1")
    plt.savefig(plot_name + ".pdf")
    plt.clf()

    if game in ['iagRNE', 'iagR', 'iagM']:
        x_axis_labels = ["L", "M"]
        y_axis_labels = ["L", "M"]
    else:
        x_axis_labels = ["L", "M", "R"]
        y_axis_labels = ["L", "M", "R"]

    df = pd.read_csv(f'{path_data}/states_{info}_{l1}_{l2}.csv', header=None)
    ax = sns.heatmap(df, annot=True, cmap="YlGnBu", vmin=0, vmax=1, xticklabels=x_axis_labels,
                     yticklabels=y_axis_labels)
    plot_name = f"{path_plots}/states"
    plt.savefig(plot_name + ".pdf")
    plt.clf()

    # action probs
    df1 = pd.read_csv(f'{path_data}/agent1_probs_{info}.csv')
    ax = sns.lineplot(x='Episode', y='Action 1', linewidth=2.0, data=df1, ci='sd',
                      label='L')
    ax = sns.lineplot(x='Episode', y='Action 2', linewidth=2.0, data=df1,
                      ci='sd', label='M')
    if game not in ['iagRNE', 'iagR', 'iagM']:
        ax = sns.lineplot(x='Episode', y='Action 3', linewidth=2.0, data=df1,
                          ci='sd', label='R')

    ax.set(ylabel='Action probability')
    ax.set(xlabel='Iterations')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, episodes)
    plot_name = f"{path_plots}/probs_ag1"
    plt.title(f"Action probabilities - Agent 1")
    plt.savefig(plot_name + ".pdf")
    plt.clf()

    df1 = pd.read_csv(f'{path_data}/agent2_probs_{info}.csv')

    ax = sns.lineplot(x='Episode', y='Action 1', linewidth=2.0, data=df1, ci='sd',
                      label='L')
    ax = sns.lineplot(x='Episode', y='Action 2', linewidth=2.0, data=df1,
                      ci='sd', label='M')
    if game not in ['iagRNE', 'iagR', 'iagM']:
        ax = sns.lineplot(x='Episode', y='Action 3', linewidth=2.0, data=df1,
                          ci='sd', label='R')

    ax.set(ylabel='Action probability')
    ax.set(xlabel='Iterations')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, episodes)
    plot_name = f"{path_plots}/probs_ag2"
    plt.title(f"Action probabilities - Agent 2")
    plt.savefig(plot_name + ".pdf")
    plt.clf()


if __name__ == "__main__":
    experiment = ['Q', 'Q']
    info = '0M'
    l1 = 1
    l2 = 1

    episodes = 5000
    moocs = ['SER']
    games = ['iag', 'iagR', 'iagM', 'iagRNE', 'iagNE'] # ['iagRNE'] # ['iag']['iagM']'iagNE',

    for l1 in range(1, 2):
        for l2 in range(1, 2):
            for mooc in moocs:
                for game in games:
                    path_data = f'results/tour_{experiment}_{game}_l{l1}_{l2}'
                    plot_results(game, mooc, path_data, experiment)
