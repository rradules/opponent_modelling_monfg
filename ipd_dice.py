# coding: utf-8

import numpy as np
import torch
import pandas as pd

from envs import IPD
from utils.hps import HpLolaDice
from agents.pg_dice import PGDiceBase, PGDice1M
from utils.utils import mkdir_p
from collections import Counter
import argparse
from envs.monfgs import get_payoff_matrix

payoff_episode_log1 = []
payoff_episode_log2 = []
act_hist_log = [[], []]


def get_return(rewards, utility):
    rewards = utility(torch.Tensor(rewards).permute(1, 2, 0))
    ret = torch.mean(rewards).item()

    return ret


def step(agent1, agent2):
    theta1 = agent1.theta
    theta2 = agent2.theta

    values1 = agent1.values
    values2 = agent2.values

    rewards1 = []
    rewards2 = []
    actions1 = []
    actions2 = []

    # just to evaluate progress:
    (s1, s2), _ = env.reset()
    for t in range(hp.len_rollout):
        a1, _, _ = agent1.act(s1, theta1, values1)
        a2, _, _ = agent2.act(s2, theta2, values2)
        (s1, s2), (r1, r2), _, _ = env.step((a1, a2))
        rewards1.append(r1)
        rewards2.append(r2)
        actions1.append(a1)
        actions2.append(a2)
    return rewards1, rewards2, actions1, actions2


def play(n_lookaheads, trials, info, mooc, game, experiment):

    state_distribution_log = np.zeros((env.NUM_ACTIONS, env.NUM_ACTIONS))
    print("start iterations with", n_lookaheads, "lookaheads:")

    for trial in range(trials):
        if info == '0M':
            agent1 = PGDiceBase(0, env, hp, u1, u2, mooc)
            agent2 = PGDiceBase(1, env, hp, u2, u1, mooc)
        else:
            agent1 = PGDice1M(0, env, hp, u1, u2, mooc)
            agent2 = PGDice1M(1, env, hp, u2, u1, mooc)

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
            if update >= (0.1 * hp.n_update):
                for rol_a in range(len(a1)):
                    for batch_a in range(len(a1[rol_a])):
                        state_distribution_log[a1[rol_a][batch_a], a2[rol_a][batch_a]] += 1

            score1 = get_return(r1, u1)
            score2 = get_return(r2, u2)

            act_probs1 = get_act_probs(a1)
            act_probs2 = get_act_probs(a2)
            act_hist_log[0].append([update, trial, n_lookaheads, act_probs1[0], act_probs1[1]])
            act_hist_log[1].append([update, trial, n_lookaheads, act_probs2[0], act_probs2[1]])

            scores = np.mean([score1, score2])
            payoff_episode_log1.append([update, trial, n_lookaheads, scores])
            payoff_episode_log2.append([update, trial, n_lookaheads, scores])

            '''
            if update % 10 == 0:
                print('update', update, 'score (%.3f,%.3f)' % (score1, score2),
                      f'probabilities (agent1) = {act_probs1[0], act_probs1[1]}',
                      f'probabilities (agent2) = {act_probs2[0], act_probs2[1]}')
            '''

    columns = ['Episode', 'Trial', 'Lookahead', 'Payoff']
    df1 = pd.DataFrame(payoff_episode_log1, columns=columns)
    df2 = pd.DataFrame(payoff_episode_log2, columns=columns)

    path_data = f'results/{experiment}_{game}' #/{mooc}/{hp.use_baseline}'
    mkdir_p(path_data)

    df1.to_csv(f'{path_data}/agent1_payoff_{info}.csv', index=False)
    df2.to_csv(f'{path_data}/agent2_payoff_{info}.csv', index=False)

    state_distribution_log /= hp.batch_size * (0.9 * hp.n_update) * trials * hp.len_rollout
    print(np.sum(state_distribution_log))
    df = pd.DataFrame(state_distribution_log)
    df.to_csv(f'{path_data}/states_{info}_{n_lookaheads}.csv', index=False, header=None)

    columns = ['Episode', 'Trial', 'Lookahead', 'Action 1', 'Action 2']
    df1 = pd.DataFrame(act_hist_log[0], columns=columns)
    df2 = pd.DataFrame(act_hist_log[1], columns=columns)

    df1.to_csv(f'{path_data}/agent1_probs_{info}.csv', index=False)
    df2.to_csv(f'{path_data}/agent2_probs_{info}.csv', index=False)

    batch_states = torch.from_numpy(np.array([0, 1, 2, 3, 4])).long()
    probs1 = torch.sigmoid(agent1.theta)[batch_states]
    probs2 = torch.sigmoid(agent2.theta)[batch_states]
    print("Probs 1: ", probs1)
    print("Probs 2: ", probs2)
    print("Values 1: ", agent1.values)
    print("Values 2: ", agent2.values)


def get_act_probs(act_ep):
    act_probs = np.zeros(env.NUM_ACTIONS)
    for actions in act_ep:
        count = Counter(actions)
        total = sum(count.values())

        for action in range(env.NUM_ACTIONS):
            act_probs[action] += count[action] / total
    act_probs = act_probs / len(act_ep)
    return act_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-trials', type=int, default=5, help="number of trials")
    parser.add_argument('-lookahead', type=int, default=4, help="number of lookaheads")
    parser.add_argument('-lr_out', type=float, default=0.2, help="lr outer loop")
    parser.add_argument('-lr_in', type=float, default=0.3, help="lr inner loop")
    parser.add_argument('-lr_v', type=float, default=0.1, help="lr values")
    parser.add_argument('-gamma', type=float, default=0.96, help="gamma")
    parser.add_argument('-updates', type=int, default=500, help="updates")
    parser.add_argument('-batch', type=int, default=128, help="batch size")
    parser.add_argument('-rollout', type=int, default=150, help="rollout size")
    parser.add_argument('-mooc', type=str, default='ESR', help="MOO criterion")
    parser.add_argument('-seed', type=int, default=42, help="seed")
    parser.add_argument('-baseline', action='store_true', help="Variance reduction")
    parser.add_argument('-no-baseline', action='store_false', help="Variance reduction")
    parser.add_argument('-game', type=str, default='ipd', help="game")
    parser.add_argument('-mem', type=str, default='1M', help="memory")
    parser.add_argument('-experiment', type=str, default='Adam', help="experiment")

    args = parser.parse_args()

    u1 = lambda x: torch.sum(torch.pow(x, 2), dim=0)
    u2 = lambda x: torch.prod(x, dim=0)

    u1 = u2 = lambda x: x[0]

    n_lookaheads = args.lookahead
    mooc = args.mooc
    seed = args.seed
    trials = args.trials
    game = args.game

    hp = HpLolaDice(args.lr_out, args.lr_in, args.lr_v, args.gamma,
                    args.updates, args.rollout, args.batch, args.baseline)

    # payout_mat = get_payoff_matrix(game)
    # iga = IGA(hp.len_rollout, hp.batch_size, payout_mat)
    env = IPD(hp.len_rollout, hp.batch_size)
    experiment = args.experiment

    info = args.mem

    for i in range(n_lookaheads):
        #torch.manual_seed(seed)
        #np.random.seed(seed)
        play(i, trials, info, mooc, game, experiment)