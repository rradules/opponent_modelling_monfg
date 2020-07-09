# coding: utf-8

import numpy as np
import torch
import pandas as pd

from envs import IGA
from utils.hp_dice import Hp
from agents.agent_dice import AgentBase, Agent1M
from utils.utils import mkdir_p
from collections import Counter
import argparse

hp = Hp()

iga = IGA(hp.len_rollout, hp.batch_size)

payoff_episode_log1 = []
payoff_episode_log2 = []
act_hist_log = [[], []]


def step(agent1, agent2):
    theta1 = agent1.theta
    theta2 = agent2.theta

    values1 = agent1.values
    values2 = agent2.values

    # just to evaluate progress:
    (s1, s2), _ = iga.reset()
    #for t in range(hp.len_rollout):
    a1, lp1, v1 = agent1.act(s1, theta1, values1)
    a2, lp2, v2 = agent2.act(s2, theta2, values2)
    (s1, s2), (r1, r2), _, _ = iga.step((a1, a2))
    return r1, r2, a1, a2


def play(agent1, agent2, n_lookaheads, trials, info, mooc):

    state_distribution_log = np.zeros((iga.action_space[0].n, iga.action_space[1].n))
    print("start iterations with", n_lookaheads, "lookaheads:")
    for trial in range(trials):
        for update in range(hp.n_update):
            # copy other's parameters:
            theta1_ = agent1.theta.clone().detach().requires_grad_(True)
            values1_ = agent1.values.clone().detach().requires_grad_(True)
            theta2_ = agent2.theta.clone().detach().requires_grad_(True)
            values2_ = agent2.values.clone().detach().requires_grad_(True)

            for k in range(n_lookaheads):
                # estimate other's gradients from in_lookahead:
                grad2 = agent1.in_lookahead(theta2_, values2_)
                grad1 = agent2.in_lookahead(theta1_, values1_)
                # update other's theta
                theta2_ = theta2_ - hp.lr_in * grad2
                theta1_ = theta1_ - hp.lr_in * grad1

            # update own parameters from out_lookahead:
            agent1.out_lookahead(theta2_, values2_)
            agent2.out_lookahead(theta1_, values1_)

            # evaluate progress:
            r1, r2, a1, a2 = step(agent1, agent2)
            for idx in range(len(a1)):
                state_distribution_log[a1[idx], a2[idx]] += 1

            act_probs1 = get_act_probs(a1)
            act_probs2 = get_act_probs(a2)
            act_hist_log[0].append([update, trial, n_lookaheads, act_probs1[0], act_probs1[1], act_probs1[2]])
            act_hist_log[1].append([update, trial, n_lookaheads, act_probs2[0], act_probs2[1], act_probs2[2]])

            payoff_episode_log1.append([update, trial, n_lookaheads, np.mean(u1(r1))])
            payoff_episode_log2.append([update, trial, n_lookaheads, np.mean(u2(r2))])

    columns = ['Episode', 'Trial', 'Lookahead', 'Payoff']
    df1 = pd.DataFrame(payoff_episode_log1, columns=columns)
    df2 = pd.DataFrame(payoff_episode_log2, columns=columns)

    path_data = f'results/dice/{mooc}'
    mkdir_p(path_data)

    df1.to_csv(f'{path_data}/agent1_payoff_{info}.csv', index=False)
    df2.to_csv(f'{path_data}/agent2_payoff_{info}.csv', index=False)

    state_distribution_log /= hp.batch_size * hp.n_update * trials
    print(np.sum(state_distribution_log))
    df = pd.DataFrame(state_distribution_log)
    df.to_csv(f'{path_data}/states_{info}_{n_lookaheads}.csv', index=False, header=None)

    columns = ['Episode', 'Trial', 'Lookahead', 'Action 1', 'Action 2', 'Action 3']
    df1 = pd.DataFrame(act_hist_log[0], columns=columns)
    df2 = pd.DataFrame(act_hist_log[1], columns=columns)

    df1.to_csv(f'{path_data}/agent1_probs_{info}.csv', index=False)
    df2.to_csv(f'{path_data}/agent2_probs_{info}.csv', index=False)


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
    parser.add_argument('-lookahead', type=int, default=4, help="number of lookaheads")
    parser.add_argument('-mooc', type=str, default='SER', help="MOO criterion")

    args = parser.parse_args()

    u1 = lambda x: x[0] ** 2 + x[1] ** 2
    u2 = lambda x: x[0] * x[1]

    info = ["0M", "1M"]
    n_lookaheads = args.lookahead
    mooc = args.mooc
    trials = args.trials
    for el in info:
        for i in range(n_lookaheads):
            torch.manual_seed(hp.seed)
            if el == '0M':
                play(AgentBase(iga, hp, u1, u2, mooc), AgentBase(iga, hp, u2, u1, mooc), i, trials, el, mooc)
            else:
                play(Agent1M(iga, hp, u1, u2, mooc), Agent1M(iga, hp, u2, u1, mooc), i, trials, el, mooc)
