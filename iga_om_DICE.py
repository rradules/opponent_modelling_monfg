# coding: utf-8
import numpy as np
import torch
import pandas as pd

from envs import IGA
from utils.hps import HpLolaDice
from agents.agent_lola_dice import AgentDiceBase, AgentDice1M
from utils.utils import mkdir_p
from collections import Counter
import argparse
from envs.monfgs import get_payoff_matrix
from copy import deepcopy

payoff_episode_log1 = []
payoff_episode_log2 = []
act_hist_log = [[], []]

def estimate_policy0M(actions):
    return get_act_probs(actions)

def estimate_values(reward):


def step0M(agent1, agent2):
    #theta1, theta2, values1, values2
    (s1, s2), _ = env.reset()
    for t in range(hp.len_rollout):
        s1_ = deepcopy(s1)
        s2_ = deepcopy(s2)
        a1, lp1, v1 = agent1.act(s1, agent1.theta, agent1.values)
        a2, lp2, v2 = agent2.act(s2, agent2.theta, agent2.values)
        (s1, s2), (r1, r2), _, _ = env.step((a1, a2))
    # one can also use r1 and r2 to estimate opponent values
    # in our case we can use our own, since it is a team reward setting

    # infer opponent's parameters
    theta1_ = -np.log(estimate_policy0M(a1))
    theta2_ = -np.log(estimate_policy0M(a2))
    theta1_ = torch.from_numpy(theta1_).float().requires_grad_()
    theta2_ = torch.from_numpy(theta2_).float().requires_grad_()

    return theta1_, theta2_


def play(agent1, agent2, n_lookaheads, trials, info, mooc, game):
    state_distribution_log = np.zeros((env.action_space[0].n, env.action_space[1].n))
    print("start iterations with", n_lookaheads, "lookaheads:")

    # init opponent models
    theta1_ = torch.zeros(env.NUM_ACTIONS, requires_grad=True)
    theta2_ = torch.zeros(env.NUM_ACTIONS, requires_grad=True)
    for update in range(hp.n_update):
        # agents use own values to model others:
        values1_ = torch.tensor(agent2.values.detach(), requires_grad=True)
        values2_ = torch.tensor(agent1.values.detach(), requires_grad=True)

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
        if update >= (0.9 * hp.n_update):
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

    path_data = f'results/{game}/{mooc}'
    mkdir_p(path_data)

    df1.to_csv(f'{path_data}/agent1_payoff_{info}.csv', index=False)
    df2.to_csv(f'{path_data}/agent2_payoff_{info}.csv', index=False)

    state_distribution_log /= hp.batch_size * (0.1 * hp.n_update) * trials
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
    act_probs = np.ones(env.NUM_ACTIONS)
    for action in range(env.action_space[0].n):
        act_probs[action] = count[action] / (total + env.NUM_ACTIONS)
    return act_probs

# plot progress:
if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-trials', type=int, default=10, help="number of trials")
    parser.add_argument('-lookahead', type=int, default=5, help="number of lookaheads")
    parser.add_argument('-mooc', type=str, default='SER', help="MOO criterion")
    parser.add_argument('-seed', type=int, default=42, help="seed")
    parser.add_argument('-game', type=str, default='iga', help="game")

    args = parser.parse_args()

    u1 = lambda x: x[0] ** 2 + x[1] ** 2
    u2 = lambda x: x[0] * x[1]

    info = ["0M", "1M"]
    n_lookaheads = args.lookahead
    mooc = args.mooc
    seed = args.seed
    trials = args.trials
    game = args.game

    hp = HpLolaDice()
    payout_mat = get_payoff_matrix(game)
    env = IGA(hp.len_rollout, hp.batch_size, payout_mat)

    for el in info:
        for i in range(n_lookaheads):
            torch.manual_seed(seed)
            np.random.seed(seed)

            if el == '0M':
                play(AgentDiceBase(env, hp, u1, u2, mooc), AgentDiceBase(env, hp, u2, u1, mooc), i, trials, el, mooc, game)
            else:
                play(AgentDice1M(env, hp, u1, u2, mooc), AgentDice1M(env, hp, u2, u1, mooc), i, trials, el, mooc, game)
