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


def plot_results(game, path_data):
    path_plots = f'plots/PGA_test/{game}'
    mkdir_p(path_plots)

    df1 = pd.read_csv(f'{path_data}/agent1_payoff.csv')

    ax = sns.lineplot(x='Episode', y='Payoff', linewidth=2.0, data=df1, ci='sd',
                          label=f'Agent 1')

    df2 = pd.read_csv(f'{path_data}/agent2_payoff.csv')

    ax = sns.lineplot(x='Episode', y='Payoff', linewidth=2.0, data=df2, ci='sd',
                          label=f'Agent 2')

    ax.set(ylabel='Scalarised payoff per step')
    ax.set(xlabel='Iterations')
    #ax.set_ylim(0, 1)
    ax.set_xlim(0, episodes)
    plot_name = f"{path_plots}/payoffs"
    plt.title("Scalarised Expected Payoffs")
    plt.savefig(plot_name + ".pdf")
    plt.clf()


    # state distribution
    if game == 'MP':
        x_axis_labels = ["1", "2"]
        y_axis_labels = ["1", "2"]
    else:
        x_axis_labels = ["1", "2", "3"]
        y_axis_labels = ["1", "2", "3"]

    df = pd.read_csv(f'{path_data}/states.csv', header=None)
    ax = sns.heatmap(df, annot=True, cmap="YlGnBu", vmin=0, vmax=1, xticklabels=x_axis_labels,
                     yticklabels=y_axis_labels)
    plot_name = f"{path_plots}/states"
    plt.savefig(plot_name + ".pdf")
    plt.clf()

    if game == "MP":
        # action probs
        df1 = pd.read_csv(f'{path_data}/agent1_probs.csv')
        ax = sns.lineplot(x='Episode', y='Action 1', linewidth=2.0, data=df1, ci='sd',
                          label='Agent 1')

        df1 = pd.read_csv(f'{path_data}/agent2_probs.csv')

        ax = sns.lineplot(x='Episode', y='Action 1', linewidth=2.0, data=df1, ci='sd',
                          label='Agent 2')

        ax.set(ylabel='Action 1 - probability')
        ax.set(xlabel='Iterations')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, episodes)
        plot_name = f"{path_plots}/probs_actions"
        plt.title(f"Probability of Action 1")
        plt.savefig(plot_name + ".pdf")
        plt.clf()
    else:
        # action probs
        df1 = pd.read_csv(f'{path_data}/agent1_probs.csv')
        ax = sns.lineplot(x='Episode', y='Action 1', linewidth=2.0, data=df1, ci='sd',
                          label='Action 1')
        ax = sns.lineplot(x='Episode', y='Action 2', linewidth=2.0, data=df1, ci='sd',
                          label='Action 2')
        ax = sns.lineplot(x='Episode', y='Action 3', linewidth=2.0, data=df1, ci='sd',
                          label='Action 3')
        ax.set(ylabel='Agent 1 - probability')
        ax.set(xlabel='Iterations')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, episodes)
        plot_name = f"{path_plots}/probs_agent1"
        plt.title(f"Probability of Agent 1")
        plt.savefig(plot_name + ".pdf")
        plt.clf()

        df1 = pd.read_csv(f'{path_data}/agent2_probs.csv')
        ax = sns.lineplot(x='Episode', y='Action 1', linewidth=2.0, data=df1, ci='sd',
                          label='Action 1')
        ax = sns.lineplot(x='Episode', y='Action 2', linewidth=2.0, data=df1, ci='sd',
                          label='Action 2')
        ax = sns.lineplot(x='Episode', y='Action 3', linewidth=2.0, data=df1, ci='sd',
                          label='Action 3')

        ax.set(ylabel='Agent 2 - probability')
        ax.set(xlabel='Iterations')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, episodes)
        plot_name = f"{path_plots}/probs_agent2"
        plt.title(f"Probability of Agent 2")
        plt.savefig(plot_name + ".pdf")
        plt.clf()


if __name__ == "__main__":

    episodes = 20000
    game = 'MP'

    path_data = f'results/PGA_test/{game}'
    plot_results(game, path_data)
