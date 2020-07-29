# coding: utf-8

import numpy as np
import torch
import pandas as pd

from envs import IGA
from utils.hps import HpPGA_APP
from agents.agent_pga_app import AgentPGAAPPBase, AgentPGAAPP1M
from utils.utils import mkdir_p
from collections import Counter
import argparse
from envs.monfgs import get_payoff_matrix

payoff_episode_log1 = []
payoff_episode_log2 = []
act_hist_log = [[], []]


def check_performance(agent1, agent2):
    (s1, s2), _ = iga.reset()
    a1 = agent1.act(s1)
    a2 = agent2.act(s2)
    (_, _), (r1, r2), _, _ = iga.step((a1, a2))

    return r1, r2, a1, a2


def play(agent1, agent2, trials, mem, mooc, game):

    state_distribution_log = np.zeros((iga.action_space[0].n, iga.action_space[1].n))
    for trial in range(trials):
        print(f'Starting trial {trial}...')

        for update in range(hp.n_update):
            (s1, s2), _ = iga.reset()
            hp.update_lr(update)
            for roll in range(hp.len_rollout):
                # step in the env (batch size)
                a1 = agent1.act(s1)
                a2 = agent2.act(s2)

                (s1, s2), (r1, r2), _, _ = iga.step((a1, a2))

                # update own parameters
                agent1.perform_update(a1, s1, r1)
                agent2.perform_update(a2, s2, r2)

            rew1, rew2, act1, act2 = check_performance(agent1, agent2)

            # record state distribution for the last 10% updates
            if update >= (0.9*hp.n_update):
                for idx in range(len(act1)):
                    state_distribution_log[act1[idx], act2[idx]] += 1

            act_probs1 = get_act_probs(act1)
            act_probs2 = get_act_probs(act2)
            act_hist_log[0].append([update, trial, act_probs1[0], act_probs1[1], act_probs1[2]])
            act_hist_log[1].append([update, trial, act_probs2[0], act_probs2[1], act_probs2[2]])

            payoff_episode_log1.append([update, trial, np.mean(u1(rew1))])
            payoff_episode_log2.append([update, trial, np.mean(u2(rew2))])

    columns = ['Episode', 'Trial', 'Payoff']
    df1 = pd.DataFrame(payoff_episode_log1, columns=columns)
    df2 = pd.DataFrame(payoff_episode_log2, columns=columns)

    path_data = f'results/PGA_APP/{game}/{mooc}'
    mkdir_p(path_data)

    df1.to_csv(f'{path_data}/agent1_payoff_{mem}.csv', index=False)
    df2.to_csv(f'{path_data}/agent2_payoff_{mem}.csv', index=False)

    state_distribution_log /= hp.batch_size * (0.1 * hp.n_update) * trials
    print(np.sum(state_distribution_log))
    df = pd.DataFrame(state_distribution_log)
    df.to_csv(f'{path_data}/states_{mem}.csv', index=False, header=None)

    columns = ['Episode', 'Trial', 'Action 1', 'Action 2', 'Action 3']
    df1 = pd.DataFrame(act_hist_log[0], columns=columns)
    df2 = pd.DataFrame(act_hist_log[1], columns=columns)

    df1.to_csv(f'{path_data}/agent1_probs_{mem}.csv', index=False)
    df2.to_csv(f'{path_data}/agent2_probs_{mem}.csv', index=False)


def get_act_probs(actions):
    count = Counter(actions)
    total = sum(count.values())
    act_probs = [0, 0, 0]
    for action in range(iga.action_space[0].n):
        act_probs[action] = count[action] / total
    return act_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-trials', type=int, default=10, help="number of trials")
    parser.add_argument('-mooc', type=str, default='SER', help="MOO criterion")
    parser.add_argument('-seed', type=int, default=42, help="seed")
    parser.add_argument('-game', type=str, default='iga', help="game")

    args = parser.parse_args()

    u1 = lambda x: x[0] ** 2.0 + x[1] ** 2.0
    u2 = lambda x: x[0] * x[1]

    info = ["0M", "1M"]
    mooc = args.mooc
    seed = args.seed
    trials = args.trials
    game = args.game

    hp = HpPGA_APP()
    payout_mat = get_payoff_matrix(game)
    iga = IGA(hp.len_rollout, hp.batch_size, payout_mat)

    for el in info:
        #torch.manual_seed(seed)
        #np.random.seed(seed)

        if el == '0M':
            play(AgentPGAAPPBase(iga, hp, u1, u2, mooc), AgentPGAAPPBase(iga, hp, u2, u1, mooc), trials, el, mooc, game)
        else:
            play(AgentPGAAPP1M(iga, hp, u1, u2, mooc), AgentPGAAPP1M(iga, hp, u2, u1, mooc), trials, el, mooc, game)
