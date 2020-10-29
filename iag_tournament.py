# coding: utf-8

import numpy as np
import torch
import pandas as pd

from envs import IAG
from utils.hps import HpLolaDice, HpAC, HpGP
from agents.pg_dice import PGDiceBase, PGDice1M, PGDiceOM
from agents.ala_ac_om import ActorCriticAgent, OppoModelingACAgent, UMOMACAgent
from utils.utils import mkdir_p
from collections import Counter
import argparse
from envs.monfgs import get_payoff_matrix

payoff_episode_log1 = []
payoff_episode_log2 = []
act_hist_log = [[], []]
trace_log = [[], []]


def get_return(rewards, utility, mooc):
    if mooc == 'SER':
        rewards = torch.mean(torch.mean(torch.Tensor(rewards).permute(1, 2, 0), dim=2), dim=1)
        ret = utility(rewards).item()
    else:
        rewards = utility(torch.mean(torch.Tensor(rewards).permute(1, 2, 0), dim=2))
        ret = torch.mean(rewards).item()
    return rewards.detach().cpu().numpy(), ret


def step(agent1, agent2):
    theta1 = agent1.theta
    theta2 = agent2.theta

    rewards1 = []
    rewards2 = []
    actions1 = []
    actions2 = []

    # just to evaluate progress:
    (s1, s2), _ = env.reset()
    for t in range(hpL.len_rollout):
        a1, _ = agent1.act(s1, theta1)
        a2, _ = agent2.act(s2, theta2)
        (s1, s2), (r1, r2), _, _ = env.step((a1, a2))
        rewards1.append(r1)
        rewards2.append(r2)
        actions1.append(a1)
        actions2.append(a2)
    return rewards1, rewards2, actions1, actions2


def AComGP_loop(agent, actions, rewards, op_actions, op_theta, lookahead):
    for k in range(lookahead):
        agent.set_op_theta(op_theta)
        if k == 0:
            umodel, likelihood = agent.makeUModel()
        agent.in_lookahead(umodel, likelihood)
    # update own parameters:
    agent.update(actions, rewards, opp_actions=op_actions)


def LOLA_loop(agent, op_theta, lookahead):
    for k in range(lookahead):
        agent.set_op_theta(op_theta)
        agent.in_lookahead()
    # update own parameters from out_lookahead:
    agent.out_lookahead()


def LOLAom_loop(agent, op_theta, lookahead):
    for k in range(lookahead):
        agent.set_op_theta(op_theta)
        if k == 0:
            umodel, likelihood = agent.makeUModel()
        agent.in_lookahead(umodel, likelihood)
    # update own parameters from out_lookahead:
    agent.out_lookahead()

def play(n_lookaheads1, n_lookaheads2, trials, info, mooc, game, experiment):
    state_distribution_log = np.zeros((env.NUM_ACTIONS, env.NUM_ACTIONS))
    print("start iterations with", n_lookaheads1, 'and', n_lookaheads2, "lookaheads:")

    for trial in range(trials):
        agents = [None, None]
        for i in range(len(experiment)):
            if experiment[i] == 'AC':
                agents[i] = ActorCriticAgent(i, hpAC, u[i], env.NUM_ACTIONS)
            elif experiment[i] == 'ACom':
                agents[i] = OppoModelingACAgent(i, hpAC, u[i], env.NUM_ACTIONS)
            elif experiment[i] == 'AComGP':
                agents[i] = UMOMACAgent(i, hpAC, u[i], env.NUM_ACTIONS, hpGP=hpGP)
            elif experiment[i] == 'LOLA':
                if info == '0M':
                    agents[i] = PGDiceBase(i, env, hpL, u[i], mooc, u[i - 1])
                else:
                    agents[i] = PGDice1M(i, env, hpL, u[i], mooc, u[i - 1])
            elif experiment[i] == 'LOLAom':
                agents[i] = PGDiceOM(i, env, hpL, u[i], mooc, hpGP=hpGP)
        agent1, agent2 = agents

        print(agent1.__class__)
        print(agent2.__class__)

        for update in range(hpL.n_update):
            # rollout actual current policies:

            if update % 100 == 0:
                print(f"Episode {update}...")
            r1, r2, a1, a2 = step(agent1, agent2)

            act_probs1 = get_act_probs(a1)
            act_probs2 = get_act_probs(a2)

            # if LOLA-LOLA
            # copy other's parameters:
            if experiment == ['LOLA', 'LOLA']:
                theta1_ = agent1.theta.clone().detach().requires_grad_(True)
                theta2_ = agent2.theta.clone().detach().requires_grad_(True)
                LOLA_loop(agent1, theta2_, n_lookaheads1)
                LOLA_loop(agent2, theta1_, n_lookaheads2)

            # if LOLAom-LOLAom
            if experiment == ['LOLAom', 'LOLAom']:
                agent1.update_logs(np.log(act_probs2))
                agent2.update_logs(np.log(act_probs1))
                if update > 1:
                    LOLAom_loop(agent1, torch.tensor(np.log(act_probs2)), n_lookaheads1)
                    LOLAom_loop(agent2, torch.tensor(np.log(act_probs1)), n_lookaheads2)

            if experiment == ['LOLAom', 'LOLA']:
                agent1.update_logs(np.log(act_probs2))
                if update > 1:
                    theta1_ = agent1.theta.clone().detach().requires_grad_(True)
                    LOLAom_loop(agent1, torch.tensor(np.log(act_probs2)), n_lookaheads1)
                    LOLA_loop(agent2, theta1_, n_lookaheads2)

            # if LOLAom-LOLAom
            if experiment == ['LOLA', 'LOLAom']:
                agent2.update_logs(np.log(act_probs1))
                if update > 1:
                    theta2_ = agent1.theta.clone().detach().requires_grad_(True)
                    LOLA_loop(agent1, theta2_, n_lookaheads1)
                    LOLAom_loop(agent2, torch.tensor(np.log(act_probs1)), n_lookaheads2)

            if experiment == ['AC', 'AC']:
                agent1.update(a1, r1)
                agent2.update(a2, r2)

            if experiment == ['AComGP', 'AComGP']:
                agent1.update_logs(act_probs2)
                agent2.update_logs(act_probs1)
                if update > 1:
                    AComGP_loop(agent1, a1, r1, a2, act_probs2, n_lookaheads1)
                    AComGP_loop(agent2, a2, r2, a1, act_probs1, n_lookaheads2)

            if experiment == ['ACom', 'LOLAom']:
                agent2.update_logs(np.log(act_probs1))
                if update > 1:
                    agent1.set_op_theta(act_probs2)
                    agent1.update(a1, r1, a2)
                    LOLAom_loop(agent2, torch.tensor(np.log(act_probs1)), n_lookaheads2)

            if experiment == ['LOLAom', 'ACom']:
                agent1.update_logs(np.log(act_probs2))
                if update > 1:
                    LOLAom_loop(agent1, torch.tensor(np.log(act_probs2)), n_lookaheads1)
                    agent2.set_op_theta(act_probs1)
                    agent2.update(a2, r2, a1)

            if experiment == ['ACom', 'ACom']:
                agent1.set_op_theta(act_probs2)
                agent1.update(a1, r1, a2)
                agent2.set_op_theta(act_probs1)
                agent2.update(a2, r2, a1)

            if experiment == ['AC', 'ACom']:
                agent1.update(a1, r1)
                agent2.set_op_theta(act_probs1)
                agent2.update(a2, r2, a1)

            if experiment == ['ACom', 'AC']:
                agent1.set_op_theta(act_probs2)
                agent1.update(a1, r1, a2)
                agent2.update(a2, r2)

            if update >= (0.1 * hpL.n_update):
                for rol_a in range(len(a1)):
                    for batch_a in range(len(a1[rol_a])):
                        state_distribution_log[a1[rol_a][batch_a], a2[rol_a][batch_a]] += 1

            ret1, score1 = get_return(r1, u1, mooc)
            ret2, score2 = get_return(r2, u2, mooc)

            if env.NUM_ACTIONS == 2:
                act_hist_log[0].append([update, trial, n_lookaheads1,
                                        act_probs1[0], act_probs1[1]])
                act_hist_log[1].append([update, trial, n_lookaheads2,
                                        act_probs2[0], act_probs2[1]])

                trace_log[0].append([update, trial, n_lookaheads1, ret1[0], ret1[1],
                                     act_probs1[0], act_probs1[1]])
                trace_log[1].append([update, trial, n_lookaheads2, ret2[0], ret2[1],
                                     act_probs2[0], act_probs2[1]])

            else:
                act_hist_log[0].append([update, trial, n_lookaheads1,
                                        act_probs1[0], act_probs1[1], act_probs1[2]])
                act_hist_log[1].append([update, trial, n_lookaheads2,
                                        act_probs2[0], act_probs2[1], act_probs2[2]])

                trace_log[0].append([update, trial, n_lookaheads1, ret1[0], ret1[1],
                                     act_probs1[0], act_probs1[1], act_probs1[2]])
                trace_log[1].append([update, trial, n_lookaheads2, ret2[0], ret2[1],
                                     act_probs2[0], act_probs2[1], act_probs2[2]])

            payoff_episode_log1.append([update, trial, n_lookaheads1, score1])
            payoff_episode_log2.append([update, trial, n_lookaheads2, score2])

    columns = ['Episode', 'Trial', 'Lookahead', 'Payoff']
    df1 = pd.DataFrame(payoff_episode_log1, columns=columns)
    df2 = pd.DataFrame(payoff_episode_log2, columns=columns)

    path_data = f'results/tour_{experiment}_{game}_l{n_lookaheads1}_{n_lookaheads2}'  # /{mooc}/{hp.use_baseline}'
    mkdir_p(path_data)

    df1.to_csv(f'{path_data}/agent1_payoff_{info}.csv', index=False)
    df2.to_csv(f'{path_data}/agent2_payoff_{info}.csv', index=False)

    state_distribution_log /= hpL.batch_size * (0.9 * hpL.n_update) * trials * hpL.len_rollout
    print(np.sum(state_distribution_log))
    df = pd.DataFrame(state_distribution_log)
    df.to_csv(f'{path_data}/states_{info}_{n_lookaheads1}_{n_lookaheads2}.csv', index=False, header=None)

    if env.NUM_ACTIONS == 3:
        columns = ['Episode', 'Trial', 'Lookahead', 'Action 1', 'Action 2', 'Action 3']
        columns1 = ['Episode', 'Trial', 'Lookahead', 'O1', 'O2', 'Action 1', 'Action 2', 'Action 3']
    else:
        columns = ['Episode', 'Trial', 'Lookahead', 'Action 1', 'Action 2']
        columns1 = ['Episode', 'Trial', 'Lookahead', 'O1', 'O2', 'Action 1', 'Action 2']
    df1 = pd.DataFrame(act_hist_log[0], columns=columns)
    df2 = pd.DataFrame(act_hist_log[1], columns=columns)

    df1.to_csv(f'{path_data}/agent1_probs_{info}.csv', index=False)
    df2.to_csv(f'{path_data}/agent2_probs_{info}.csv', index=False)

    df1 = pd.DataFrame(trace_log[0], columns=columns1)
    df2 = pd.DataFrame(trace_log[1], columns=columns1)

    df1.to_csv(f'{path_data}/agent1_traces_{info}.csv', index=False)
    df2.to_csv(f'{path_data}/agent2_traces_{info}.csv', index=False)
    del df1, df2


def get_act_probs(act_ep):
    act_probs = 1e-8 * np.ones(env.NUM_ACTIONS)
    for actions in act_ep:
        count = Counter(actions)
        total = sum(count.values())

        for action in range(env.NUM_ACTIONS):
            act_probs[action] += count[action] / total
    act_probs = act_probs / len(act_ep)
    return act_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-trials', type=int, default=100, help="number of trials")
    parser.add_argument('-updates', type=int, default=3000, help="updates")
    parser.add_argument('-batch', type=int, default=1, help="batch size")
    #TODO remove the rollout, since it will always be 1
    parser.add_argument('-rollout', type=int, default=1, help="rollout size")
    parser.add_argument('-mooc', type=str, default='SER', help="MOO criterion")
    parser.add_argument('-seed', type=int, default=42, help="seed")

    # LOLA Agent
    parser.add_argument('-lr_out', type=float, default=0.1, help="lr outer loop")
    parser.add_argument('-lr_in', type=float, default=0.2, help="lr inner loop")
    parser.add_argument('-gammaL', type=float, default=1, help="gamma")
    parser.add_argument('-mem', type=str, default='0M', help="memory")

    # AC agent
    parser.add_argument('-lr_q', type=float, default=0.05, help="lr q")
    parser.add_argument('-lr_theta', type=float, default=0.05, help="lr theta")
    parser.add_argument('-gammaAC', type=float, default=1, help="gamma")

    parser.add_argument('-game', type=str, default='iagNE', help="game")
    parser.add_argument('-experiment', type=str, default='LOLAom-ACom', help="experiment")

    parser.add_argument('-lookahead1', type=int, default=1, help="number of lookaheads for agent 1")
    parser.add_argument('-lookahead2', type=int, default=1, help="number of lookaheads for agent 2")

    args = parser.parse_args()

    u1 = lambda x: torch.sum(torch.pow(x, 2), dim=0)
    u2 = lambda x: torch.prod(x, dim=0)

    u = [u1, u2]

    n_lookaheads1 = args.lookahead1
    n_lookaheads2 = args.lookahead1
    mooc = args.mooc
    seed = args.seed
    # TODO: set seed properly
    trials = args.trials
    game = args.game

    hpL = HpLolaDice(args.lr_out, args.lr_in, args.gammaL,
                     args.updates, args.rollout, args.batch)
    hpAC = HpAC(args.lr_q, args.lr_theta, args.gammaAC,
                args.updates, args.rollout, args.batch)

    hpGP = HpGP()

    payout_mat = get_payoff_matrix(game)
    print(payout_mat)
    env = IAG(hpL.len_rollout, hpL.batch_size, payout_mat)
    experiment = (args.experiment).split("-")
    print(experiment)

    info = args.mem

    print(args.lr_q, args.lr_theta)


    # for i in range(n_lookaheads):
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    play(n_lookaheads1, n_lookaheads2, trials, info, mooc, game, experiment)
