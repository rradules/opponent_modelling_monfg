import pandas as pd
import matplotlib

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

info = ["0M"] #, "1M"]
trials = 10

episodes = 1000
mooc = 'SER'
game = 'igaNE'
path_data = f'results/PAG_APP/{game}/{mooc}'

for el in info:

    path_plots = f'plots/PAG_APP/{game}/{mooc}/{el}'
    mkdir_p(path_plots)

    df1 = pd.read_csv(f'{path_data}/agent1_payoff_{el}.csv')

    ax = sns.lineplot(x='Episode', y='Payoff', linewidth=2.0, data=df1, ci='sd',
                          label=f'Agent 1')

    df2 = pd.read_csv(f'{path_data}/agent2_payoff_{el}.csv')

    ax = sns.lineplot(x='Episode', y='Payoff', linewidth=2.0, data=df2, ci='sd',
                          label=f'Agent 2')

    ax.set(ylabel='Scalarised payoff per step')
    ax.set(xlabel='Iterations')
    #ax.set_ylim(0, 4)
    ax.set_xlim(0, episodes)
    plot_name = f"{path_plots}/payoffs_{el}"
    plt.title("Scalarised Expected Payoffs")
    plt.savefig(plot_name + ".pdf")
    plt.clf()


    # state distribution
    x_axis_labels = ["L", "M", "R"]
    y_axis_labels = ["L", "M", "R"]

    df = pd.read_csv(f'{path_data}/states_{el}.csv', header=None)
    ax = sns.heatmap(df, annot=True, cmap="YlGnBu", vmin=0, vmax=1, xticklabels=x_axis_labels,
                     yticklabels=y_axis_labels)
    plot_name = f"{path_plots}/states_{el}"
    plt.savefig(plot_name + ".pdf")
    plt.clf()

    # action probs
    df1 = pd.read_csv(f'{path_data}/agent1_probs_{el}.csv')
    ax = sns.lineplot(x='Episode', y='Action 1', linewidth=2.0, data=df1, ci='sd',
                      label='L')
    ax = sns.lineplot(x='Episode', y='Action 2', linewidth=2.0, data=df1,
                      ci='sd', label='M')
    ax = sns.lineplot(x='Episode', y='Action 3', linewidth=2.0, data=df1,
                      ci='sd', label='R')

    ax.set(ylabel='Action probability')
    ax.set(xlabel='Iterations')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, episodes)
    plot_name = f"{path_plots}/probs_ag1_{el}"
    plt.title(f"Agent 1")
    plt.savefig(plot_name + ".pdf")
    plt.clf()

    df1 = pd.read_csv(f'{path_data}/agent2_probs_{el}.csv')

    ax = sns.lineplot(x='Episode', y='Action 1', linewidth=2.0, data=df1, ci='sd',
                      label='L')
    ax = sns.lineplot(x='Episode', y='Action 2', linewidth=2.0, data=df1,
                      ci='sd', label='M')
    ax = sns.lineplot(x='Episode', y='Action 3', linewidth=2.0, data=df1,
                      ci='sd', label='R')

    ax.set(ylabel='Action probability')
    ax.set(xlabel='Iterations')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, episodes)
    plot_name = f"{path_plots}/probs_ag2_{el}"
    plt.title(f"Agent 2")
    plt.savefig(plot_name + ".pdf")
    plt.clf()
