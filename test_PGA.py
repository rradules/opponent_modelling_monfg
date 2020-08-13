# coding: utf-8

import numpy as np
import pandas as pd

from envs import PGA
from utils.hps import HpPGA_APP_test
from agents.pga_app_basic import PGAAPP
from utils.utils import mkdir_p
import argparse
from envs.nfgs import get_payoff_matrix

payoff_episode_log1 = []
payoff_episode_log2 = []
act_hist_log = [[], []]


def play(agent1, agent2, trials, game):

    state_distribution_log = np.zeros((env.action_space[0].n, env.action_space[1].n))
    for trial in range(trials):
        print(f'Starting trial {trial}...')

        for update in range(hp.n_update):
            env.reset()
            hp.update_lr(update)
            # step in the env (batch size)
            a1 = agent1.act()
            a2 = agent2.act()

            (r1, r2), _, _ = env.step((a1, a2))

            # update own parameters
            agent1.perform_update(a1, r1)
            agent2.perform_update(a2, r2)

            # rew1, rew2, act1, act2 = check_performance(agent1, agent2)
            # record state distribution for the last 10% updates
            rew1, rew2, act1, act2 = r1, r2, a1, a2

            if update >= (0.9*hp.n_update):
                state_distribution_log[act1, act2] += 1

            act_probs1 = agent1.pi
            act_probs2 = agent2.pi
            if game == "MP":
                act_hist_log[0].append([update, trial, act_probs1[0], act_probs1[1]])
                act_hist_log[1].append([update, trial, act_probs2[0], act_probs2[1]])
            else:
                act_hist_log[0].append([update, trial, act_probs1[0], act_probs1[1], act_probs1[2]])
                act_hist_log[1].append([update, trial, act_probs2[0], act_probs2[1], act_probs2[2]])

            payoff_episode_log1.append([update, trial, rew1])
            payoff_episode_log2.append([update, trial, rew2])

    columns = ['Episode', 'Trial', 'Payoff']
    df1 = pd.DataFrame(payoff_episode_log1, columns=columns)
    df2 = pd.DataFrame(payoff_episode_log2, columns=columns)

    path_data = f'results/PGA_test/{game}'
    mkdir_p(path_data)

    df1.to_csv(f'{path_data}/agent1_payoff.csv', index=False)
    df2.to_csv(f'{path_data}/agent2_payoff.csv', index=False)

    state_distribution_log /= (0.1 * hp.n_update) * trials
    print(np.sum(state_distribution_log))
    df = pd.DataFrame(state_distribution_log)
    df.to_csv(f'{path_data}/states.csv', index=False, header=None)

    if game == "MP":
        columns = ['Episode', 'Trial', 'Action 1', 'Action 2']
    else:
        columns = ['Episode', 'Trial', 'Action 1', 'Action 2', 'Action 3']
    df1 = pd.DataFrame(act_hist_log[0], columns=columns)
    df2 = pd.DataFrame(act_hist_log[1], columns=columns)

    df1.to_csv(f'{path_data}/agent1_probs.csv', index=False)
    df2.to_csv(f'{path_data}/agent2_probs.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-trials', type=int, default=10, help="number of trials")
    parser.add_argument('-seed', type=int, default=42, help="seed")
    parser.add_argument('-game', type=str, default='MP', help="game")

    args = parser.parse_args()
    seed = args.seed
    trials = args.trials
    game = args.game

    payoff = get_payoff_matrix(game)
    if game == 'MP':
        init1, init2 = [0.1, 0.9], [0.9, 0.1]
    else:
        init1, init2 = [0.1, 0.8, 0.1], [0.8, 0.1, 0.1]

    hp = HpPGA_APP_test()
    env = PGA(hp.len_rollout, payoff)
    # np.random.seed(seed)
    play(PGAAPP(env, hp, init1), PGAAPP(env, hp, init2), trials, game)
