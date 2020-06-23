# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from envs import IGA
from utils.hp_dice import Hp
from agents.agent_dice import Agent

hp = Hp()

iga = IGA(hp.len_rollout, hp.batch_size)


def step(agent1, agent2):
    theta1 = agent1.theta
    theta2 = agent2.theta

    values1 = agent1.values
    values2 = agent2.values

    # just to evaluate progress:
    (s1, s2), _ = iga.reset()
    for t in range(hp.len_rollout):
        a1, lp1, v1 = agent1.act(s1, theta1, values1)
        a2, lp2, v2 = agent2.act(s2, theta2, values2)
        (s1, s2), (r1, r2), _, _ = iga.step((a1, a2))
        # cumulate scores
        # score1 += np.mean(r1[0])/float(hp.len_rollout)
        # score2 += np.mean(r2[0])/float(hp.len_rollout)
    return r1, r2, a1, a2




def play(agent1, agent2, n_lookaheads):
    payoff_episode_log1 = []
    payoff_episode_log2 = []
    state_distribution_log = np.zeros((iga.action_space[0].n, iga.action_space[1].n))
    act_hist_log = [[], []]
    final_policy_log = [[], []]

    print("start iterations with", n_lookaheads, "lookaheads:")
    for update in range(hp.n_update):
        # copy other's parameters:
        theta1_ = torch.tensor(agent1.theta.detach(), requires_grad=True)
        values1_ = torch.tensor(agent1.values.detach(), requires_grad=True)
        theta2_ = torch.tensor(agent2.theta.detach(), requires_grad=True)
        values2_ = torch.tensor(agent2.values.detach(), requires_grad=True)

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
        payoff_episode_log1.append([update, 0, np.mean(u1(r1))])
        payoff_episode_log2.append([update, 0, np.mean(u2(r2))])

        print(a1, a2)
        #state_distribution_log[selected_actions[0], selected_actions[1]] += 1

        # print
        if update % 10 == 0:
            p1 = [p.item() for p in torch.sigmoid(agent1.theta)]
            p2 = [p.item() for p in torch.sigmoid(agent2.theta)]
            print(f'policy (agent1) = {p1}', f' (agent2) = {p2}')

    return r1


# plot progress:
if __name__ == "__main__":

    colors = ['b', 'c', 'm', 'r']

    u1 = lambda x: x[0] ** 2 + x[1] ** 2
    u2 = lambda x: x[0] * x[1]

    for i in range(4):
        torch.manual_seed(hp.seed)
        scores = play(Agent(iga, hp, u1, u2), Agent(iga, hp, u2, u1), i)
        plt.plot(scores, colors[i], label=str(i) + " lookaheads")

    plt.legend()
    plt.xlabel('rollouts', fontsize=20)
    plt.ylabel('joint score', fontsize=20)
    plt.show()
