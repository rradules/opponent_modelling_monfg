# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from copy import deepcopy

from utils.hps import Hp
from envs import IGA

hp = Hp()
iga = IGA(hp.len_rollout, hp.batch_size)


def step(theta1, theta2, values1, values2):
    (s1, s2), _ = ipd.reset()
    score1 = 0
    score2 = 0
    freq1 = np.ones(5)
    freq2 = np.ones(5)
    n1 = 2*np.ones(5)
    n2 = 2*np.ones(5)
    for t in range(hp.len_rollout):
        s1_ = deepcopy(s1)
        s2_ = deepcopy(s2)
        a1, lp1, v1 = act(s1, theta1, values1)
        a2, lp2, v2 = act(s2, theta2, values2)
        (s1, s2), (r1, r2),_,_ = iga.step((a1, a2))
        # cumulate scores
        score1 += np.mean(r1)/float(hp.len_rollout)
        score2 += np.mean(r2)/float(hp.len_rollout)
        # count actions
        for i,s in enumerate(s1_):
            freq1[int(s)] += (s2==3).astype(float)[i]
            freq1[int(s)] += (s2==1).astype(float)[i]
            n1[int(s)] += 1

        for i,s in enumerate(s2_):
            freq2[int(s)] += (s1==3).astype(float)[i]
            freq2[int(s)] += (s1==1).astype(float)[i]
            n2[int(s)] += 1

    # infer opponent's parameters
    theta1_ = -np.log(n1/freq1 - 1)
    theta2_ = -np.log(n2/freq2 - 1)
    theta1_ = torch.from_numpy(theta1_).float().requires_grad_()
    theta2_ = torch.from_numpy(theta2_).float().requires_grad_()

    return (score1, score2), theta1_, theta2_


def play(agent1, agent2, n_lookaheads):
    joint_scores = []
    print("start iterations with", n_lookaheads, "lookaheads:")
    # init opponent models
    theta1_ = torch.zeros(5, requires_grad=True)
    theta2_ = torch.zeros(5, requires_grad=True)
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
        score, theta1_, theta2_ = step(agent1.theta, agent2.theta, agent1.values, agent2.values)
        joint_scores.append(0.5*(score[0] + score[1]))

        # print
        if update%10==0 :
            p1 = [p.item() for p in torch.sigmoid(agent1.theta)]
            p2 = [p.item() for p in torch.sigmoid(agent2.theta)]
            print('update', update, 'score (%.3f,%.3f)' % (score[0], score[1]) , 'policy (agent1) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (p1[0], p1[1], p1[2], p1[3], p1[4]),' (agent2) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (p2[0], p2[1], p2[2], p2[3], p2[4]))

    return joint_scores

# plot progress:
if __name__=="__main__":

    colors = ['b','c','m','r']

    for i in range(4):
        torch.manual_seed(hp.seed)
        scores = play(Agent(), Agent(), i)
        plt.plot(scores, colors[i], label=str(i)+" lookaheads")

    plt.legend()
    plt.xlabel('rollouts', fontsize=20)
    plt.ylabel('joint score', fontsize=20)
    plt.show()
