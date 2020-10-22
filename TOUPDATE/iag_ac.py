# coding: utf-8

import numpy as np
import torch
import pandas as pd

from envs import IAG
from utils.hps import HpAC
from agents.ala_ac_om import ActorCriticAgent
from utils.utils import mkdir_p
from collections import Counter
import argparse
from envs.monfgs import get_payoff_matrix

payoff_episode_log1 = []
payoff_episode_log2 = []
act_hist_log = [[], []]


def get_return(rewards, utility, mooc):
    if mooc == 'SER':
        rewards = torch.mean(torch.mean(torch.Tensor(rewards).permute(1, 2, 0), dim=2), dim=1)
        ret = utility(rewards).item()
    else:
        rewards = utility(torch.mean(torch.Tensor(rewards).permute(1, 2, 0), dim=2))
        ret = torch.mean(rewards)
    return ret


def step(agent1, agent2):
    rewards1 = []
    rewards2 = []
    actions1 = []
    actions2 = []

    # just to evaluate progress:
    (s1, s2), _ = env.reset()
    for t in range(hp.len_rollout):
        a1 = agent1.act()
        a2 = agent2.act()
        (s1, s2), (r1, r2), _, _ = env.step((a1, a2))
        rewards1.append(r1)
        rewards2.append(r2)
        actions1.append(a1)
        actions2.append(a2)
    return rewards1, rewards2, actions1, actions2


def play(trials, mooc, game, experiment):

    state_distribution_log = np.zeros((env.NUM_ACTIONS, env.NUM_ACTIONS))

    for trial in range(trials):
        print(f"Trial: {trial}")
        agent1 = ActorCriticAgent(0, hp, u1, env.NUM_ACTIONS)
        agent2 = ActorCriticAgent(1, hp, u2, env.NUM_ACTIONS)

        for update in range(hp.n_update):

            rewards1, rewards2, actions1, actions2 = step(agent1, agent2)

            # update own parameters from out_lookahead:
            agent1.update(actions1, rewards1)
            agent2.update(actions2, rewards2)

            # evaluate progress:
            r1, r2, a1, a2 = step(agent1, agent2)
            if update >= (0.1 * hp.n_update):
                for rol_a in range(len(a1)):
                    for batch_a in range(len(a1[rol_a])):
                        state_distribution_log[a1[rol_a][batch_a], a2[rol_a][batch_a]] += 1

            score1 = get_return(r1, u1, mooc)
            score2 = get_return(r2, u2, mooc)

            act_probs1 = get_act_probs(a1)
            act_probs2 = get_act_probs(a2)
            act_hist_log[0].append([update, trial,
                                    act_probs1[0], act_probs1[1], act_probs1[2]])
            act_hist_log[1].append([update, trial,
                                    act_probs2[0], act_probs2[1], act_probs2[2]])

            payoff_episode_log1.append([update, trial, score1])
            payoff_episode_log2.append([update, trial, score2])

            #print(act_probs1[0], act_probs1[1], act_probs1[2])
            #print(act_probs2[0], act_probs2[1], act_probs2[2])
            #print(score1, score2)
    columns = ['Episode', 'Trial', 'Payoff']
    df1 = pd.DataFrame(payoff_episode_log1, columns=columns)
    df2 = pd.DataFrame(payoff_episode_log2, columns=columns)

    path_data = f'results/AC_{experiment}_{game}' #/{mooc}/{hp.use_baseline}'
    mkdir_p(path_data)

    df1.to_csv(f'{path_data}/agent1_payoff.csv', index=False)
    df2.to_csv(f'{path_data}/agent2_payoff.csv', index=False)

    state_distribution_log /= hp.batch_size * (0.9 * hp.n_update) * trials * hp.len_rollout
    print(np.sum(state_distribution_log))
    df = pd.DataFrame(state_distribution_log)
    df.to_csv(f'{path_data}/states.csv', index=False, header=None)

    columns = ['Episode', 'Trial', 'Action 1', 'Action 2', 'Action 3']
    df1 = pd.DataFrame(act_hist_log[0], columns=columns)
    df2 = pd.DataFrame(act_hist_log[1], columns=columns)

    df1.to_csv(f'{path_data}/agent1_probs.csv', index=False)
    df2.to_csv(f'{path_data}/agent2_probs.csv', index=False)


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

    parser.add_argument('-trials', type=int, default=10, help="number of trials")
    parser.add_argument('-lr_q', type=float, default=0.1, help="lr q")
    parser.add_argument('-lr_theta', type=float, default=0.1, help="lr theta")
    parser.add_argument('-gamma', type=float, default=0.9, help="gamma")
    parser.add_argument('-updates', type=int, default=3000, help="updates")
    parser.add_argument('-batch', type=int, default=64, help="batch size")
    parser.add_argument('-rollout', type=int, default=150, help="rollout size")
    parser.add_argument('-mooc', type=str, default='SER', help="MOO criterion")
    parser.add_argument('-seed', type=int, default=42, help="seed")
    parser.add_argument('-game', type=str, default='iagNE', help="game")
    parser.add_argument('-experiment', type=str, default='testNE', help="experiment")

    args = parser.parse_args()

    u1 = lambda x: torch.sum(torch.pow(x, 2), dim=0)
    u2 = lambda x: torch.prod(x, dim=0)

    mooc = args.mooc
    seed = args.seed
    trials = args.trials
    game = args.game

    hp = HpAC(args.lr_q, args.lr_theta, args.gamma,
                    args.updates, args.rollout, args.batch)

    payout_mat = get_payoff_matrix(game)
    print(payout_mat)
    env = IAG(hp.len_rollout, hp.batch_size, payout_mat)
    #env = IPD(hp.len_rollout, hp.batch_size)
    experiment = args.experiment

    play(trials, mooc, game, experiment)
