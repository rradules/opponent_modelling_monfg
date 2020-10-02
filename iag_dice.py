# coding: utf-8

import numpy as np
import torch
import pandas as pd

from envs import IAG
from utils.hps import HpLolaDice
from agents.pg_dice import PGDiceBase, PGDice1M
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
        ret = torch.mean(rewards).item()
    return ret


def step(agent1, agent2):
    theta1 = agent1.theta
    theta2 = agent2.theta

    rewards1 = []
    rewards2 = []
    actions1 = []
    actions2 = []

    # just to evaluate progress:
    (s1, s2), _ = env.reset()
    for t in range(hp.len_rollout):
        a1, _ = agent1.act(s1, theta1)
        a2, _ = agent2.act(s2, theta2)
        (s1, s2), (r1, r2), _, _ = env.step((a1, a2))
        rewards1.append(r1)
        rewards2.append(r2)
        actions1.append(a1)
        actions2.append(a2)
    return np.array(rewards1), np.array(rewards2), np.array(actions1), np.array(actions2)


def play(n_lookaheads, trials, info, mooc, game, experiment):

    path_data = f'results/lola_{experiment}_{game}' #/{mooc}/{hp.use_baseline}'
    mkdir_p(path_data)

    state_distribution_log = np.zeros((env.NUM_ACTIONS, env.NUM_ACTIONS))
    print("start iterations with", n_lookaheads, "lookaheads:")

    for trial in range(trials):
        if info == '0M':
            agent1 = PGDiceBase(0, env, hp, u1, u2, mooc)
            agent2 = PGDiceBase(1, env, hp, u2, u1, mooc)
        else:
            agent1 = PGDice1M(0, env, hp, u1, u2, mooc)
            agent2 = PGDice1M(1, env, hp, u2, u1, mooc)

        trace = []

        for update in range(hp.n_update):
            # copy other's parameters:
            theta1_ = agent1.theta.clone().detach().requires_grad_(True)
            theta2_ = agent2.theta.clone().detach().requires_grad_(True)

            for k in range(n_lookaheads):
                # estimate other's gradients from in_lookahead:
                grad2 = agent1.in_lookahead(theta2_)
                grad1 = agent2.in_lookahead(theta1_)
                # update other's theta
                theta2_ = theta2_ - hp.lr_in * grad2
                theta1_ = theta1_ - hp.lr_in * grad1

            # update own parameters from out_lookahead:
            agent1.out_lookahead(theta2_)
            agent2.out_lookahead(theta1_)

            # evaluate progress:
            r1, r2, a1, a2 = step(agent1, agent2)

            act_probs1 = get_act_probs(a1)
            act_probs2 = get_act_probs(a2)

            df_a1 = pd.DataFrame(a1).stack().reset_index()
            df_r1_o1 = pd.DataFrame(r1[:, 0, :]).stack().reset_index()
            df_r1_o2 = pd.DataFrame(r1[:, 1, :]).stack().reset_index()
            df_r2_o1 = pd.DataFrame(r2[:, 0, :]).stack().reset_index()
            df_r2_o2 = pd.DataFrame(r2[:, 1, :]).stack().reset_index()
            df_a2 = pd.DataFrame(a2).stack().reset_index()
            #df_a1.rename(columns=['Rollout', 'Batch', 'Action 1'], inplace=True)
            df_a1.columns = ['Rollout', 'Batch', 'A1']
            df_a2.columns = ['Rollout', 'Batch', 'A2']
            df_r1_o1.columns = ['Rollout', 'Batch', 'R1O1']
            df_r1_o2.columns = ['Rollout', 'Batch', 'R1O2']
            df_r2_o1.columns = ['Rollout', 'Batch', 'R2O1']
            df_r2_o2.columns = ['Rollout', 'Batch', 'R2O2']

            df_trace = pd.concat([df_a1, df_r1_o1[['R1O1']], df_r1_o2[['R1O2']],
                            df_a2[['A2']], df_r2_o1[['R2O1']], df_r2_o2[['R2O2']]], axis=1, sort=False)

            df_trace['Episode'] = update
            df_trace = df_trace[['Episode', 'Rollout', 'Batch', 'A1', 'R1O1', 'R1O2', 'A2', 'R2O1', 'R2O2']]

            header = True
            if update > 0:
                header = False
            df_trace.to_csv(f'{path_data}/traces_{info}.csv', index=False, mode='a', header=header)
            del df_trace

            if update >= (0.1 * hp.n_update):
                for rol_a in range(len(a1)):
                    for batch_a in range(len(a1[rol_a])):
                        state_distribution_log[a1[rol_a][batch_a], a2[rol_a][batch_a]] += 1

            score1 = get_return(r1, u1, mooc)
            score2 = get_return(r2, u2, mooc)

            if env.NUM_ACTIONS == 2:
                act_hist_log[0].append([update, trial, n_lookaheads,
                                        act_probs1[0], act_probs1[1]])
                act_hist_log[1].append([update, trial, n_lookaheads,
                                        act_probs2[0], act_probs2[1]])
            else:
                act_hist_log[0].append([update, trial, n_lookaheads,
                                        act_probs1[0], act_probs1[1], act_probs1[2]])
                act_hist_log[1].append([update, trial, n_lookaheads,
                                        act_probs2[0], act_probs2[1], act_probs2[2]])

            payoff_episode_log1.append([update, trial, n_lookaheads, score1])
            payoff_episode_log2.append([update, trial, n_lookaheads, score2])

    columns = ['Episode', 'Trial', 'Lookahead', 'Payoff']
    df1 = pd.DataFrame(payoff_episode_log1, columns=columns)
    df2 = pd.DataFrame(payoff_episode_log2, columns=columns)


    df1.to_csv(f'{path_data}/agent1_payoff_{info}.csv', index=False)
    df2.to_csv(f'{path_data}/agent2_payoff_{info}.csv', index=False)

    del df1, df2

    state_distribution_log /= hp.batch_size * (0.9 * hp.n_update) * trials * hp.len_rollout
    print(np.sum(state_distribution_log))
    df = pd.DataFrame(state_distribution_log)
    df.to_csv(f'{path_data}/states_{info}_{n_lookaheads}.csv', index=False, header=None)

    del df

    if env.NUM_ACTIONS == 3:
        columns = ['Episode', 'Trial', 'Lookahead', 'Action 1', 'Action 2', 'Action 3']
    else:
        columns = ['Episode', 'Trial', 'Lookahead', 'Action 1', 'Action 2']
    df1 = pd.DataFrame(act_hist_log[0], columns=columns)
    df2 = pd.DataFrame(act_hist_log[1], columns=columns)

    df1.to_csv(f'{path_data}/agent1_probs_{info}.csv', index=False)
    df2.to_csv(f'{path_data}/agent2_probs_{info}.csv', index=False)

    del df1, df2

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

    parser.add_argument('-trials', type=int, default=1, help="number of trials")
    parser.add_argument('-lookahead', type=int, default=1, help="number of lookaheads")
    parser.add_argument('-lr_out', type=float, default=0.2, help="lr outer loop")
    parser.add_argument('-lr_in', type=float, default=0.3, help="lr inner loop")
    parser.add_argument('-lr_v', type=float, default=0.1, help="lr values")
    parser.add_argument('-gamma', type=float, default=0.96, help="gamma")
    parser.add_argument('-updates', type=int, default=50, help="updates")
    parser.add_argument('-batch', type=int, default=64, help="batch size")
    parser.add_argument('-rollout', type=int, default=150, help="rollout size")
    parser.add_argument('-mooc', type=str, default='SER', help="MOO criterion")
    parser.add_argument('-seed', type=int, default=42, help="seed")
    parser.add_argument('-baseline', action='store_true', help="Variance reduction")
    parser.add_argument('-no-baseline', action='store_false', help="Variance reduction")
    parser.add_argument('-game', type=str, default='iag', help="game")
    parser.add_argument('-mem', type=str, default='0M', help="memory")
    parser.add_argument('-experiment', type=str, default='trace-test', help="experiment")

    args = parser.parse_args()

    u1 = lambda x: torch.sum(torch.pow(x, 2), dim=0)
    u2 = lambda x: torch.prod(x, dim=0)

    n_lookaheads = args.lookahead
    mooc = args.mooc
    seed = args.seed
    trials = args.trials
    game = args.game

    hp = HpLolaDice(args.lr_out, args.lr_in, args.lr_v, args.gamma,
                    args.updates, args.rollout, args.batch, args.baseline)

    payout_mat = get_payoff_matrix(game)
    print(payout_mat)
    env = IAG(hp.len_rollout, hp.batch_size, payout_mat)
    #env = IPD(hp.len_rollout, hp.batch_size)
    experiment = args.experiment

    info = args.mem

    for i in range(n_lookaheads):
        #torch.manual_seed(seed)
        #np.random.seed(seed)
        play(i, trials, info, mooc, game, experiment)
