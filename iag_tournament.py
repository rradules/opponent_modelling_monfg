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
#trace_log = [[], []]


def get_return(rewards, utility, mooc):
    if mooc == 'SER':
        rewards = torch.mean(torch.mean(torch.Tensor(rewards).permute(1, 2, 0), dim=2), dim=1)
        ret = utility(rewards).item()
    else:
        rewards = utility(torch.mean(torch.Tensor(rewards).permute(1, 2, 0), dim=2))
        ret = torch.mean(rewards).item()
    return rewards.detach().cpu().numpy(), ret


def step(agents, rollout):
    theta1 = agents[0].theta
    theta2 = agents[1].theta

    rewards1 = []
    rewards2 = []
    actions1 = []
    actions2 = []

    # just to evaluate progress:
    (s1, s2), _ = env.reset()
    for t in range(rollout):
        a1, _ = agents[0].act(s1, theta1)
        a2, _ = agents[1].act(s2, theta2)
        (s1, s2), (r1, r2), _, _ = env.step((a1, a2))
        rewards1.append(r1)
        rewards2.append(r2)
        actions1.append(a1)
        actions2.append(a2)
    return [rewards1, rewards2], [actions1, actions2]


def AComGP_loop(agent, actions, rewards, op_actions, op_theta, lookahead):
    agent.set_op_theta(op_theta)
    for k in range(lookahead):
        #print(f'Lookahead loop {k}')
        if k == 0:
            umodel, likelihood = agent.makeUModel()
        agent.in_lookahead(umodel, likelihood)
    # update own parameters:
    agent.update(actions, rewards, opp_actions=op_actions)


def LOLA_loop(agent, op_theta, lookahead):
    agent.set_op_theta(op_theta)
    for k in range(lookahead):
        agent.in_lookahead()
    # update own parameters from out_lookahead:
    agent.out_lookahead()


def LOLAom_loop(agent, op_theta, lookahead):
    agent.set_op_theta(op_theta)
    for k in range(lookahead):
        if k == 0:
            umodel, likelihood = agent.makeUModel()
        agent.in_lookahead(umodel, likelihood)
    # update own parameters from out_lookahead:
    agent.out_lookahead()


def play(n_lookaheads, trials, info, mooc, game, experiment):
    state_distribution_log = np.zeros((env.NUM_ACTIONS, env.NUM_ACTIONS))
    print("start iterations with", n_lookaheads[0], 'and', n_lookaheads[1], "lookaheads:")

    for trial in range(trials):
        if trial % 10 == 0:
            print(f"Trial {trial}...")
        agents = [None, None]
        for i in range(len(experiment)):
            if experiment[i] == 'AC':
                agents[i] = ActorCriticAgent(i, hpAC, u[i], env.NUM_ACTIONS)
            elif experiment[i] == 'ACom' or experiment[i] == 'ACoa':
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
        #agent1, agent2 = agents

        #print(agents[0].__class__)
        #print(agents[1].__class__)

        for update in range(hpL.n_update):
            # rollout actual current policies:

            if update % 100 == 0:
                print(f"Episode {update}...")
            r_s, a_s = step(agents, 100)
            act_probs = [get_act_probs(a_s[0]), get_act_probs(a_s[1])]
            r, a = step(agents, 1)

            if experiment == ['LOLA', 'LOLA']:
                theta1_ = agents[0].theta.clone().detach().requires_grad_(True)
                theta2_ = agents[1].theta.clone().detach().requires_grad_(True)
                LOLA_loop(agents[0], theta2_, n_lookaheads[0])
                LOLA_loop(agents[1], theta1_, n_lookaheads[1])

            if experiment == ['ACoa', 'ACoa']:
                theta1_ = agents[0].policy
                theta2_ = agents[1].policy
                agents[0].set_op_theta(theta2_)
                agents[1].set_op_theta(theta1_)
                agents[0].update(a[0], r[0], a[1])
                agents[1].update(a[1], r[1], a[0])

            if experiment == ['LOLA', 'ACoa']:
                theta1_ = torch.sigmoid(agents[0].theta.clone().detach())
                theta2_ = torch.tensor(agents[1].policy).requires_grad_(True)
                LOLA_loop(agents[0], theta2_, n_lookaheads[0])
                agents[1].set_op_theta(theta1_.numpy())
                agents[1].update(a[1], r[1], a[0])

            if experiment == ['ACoa', 'LOLA']:
                theta1_ = torch.tensor(agents[0].policy).requires_grad_(True)
                theta2_ = torch.sigmoid(agents[1].theta.clone().detach())
                LOLA_loop(agents[1], theta1_, n_lookaheads[1])
                agents[0].set_op_theta(theta2_.numpy())
                agents[0].update(a[0], r[0], a[1])

            if experiment == ['LOLA', 'AC']:
                theta2_ = torch.tensor(agents[1].policy).requires_grad_(True)
                LOLA_loop(agents[0], theta2_, n_lookaheads[0])
                agents[1].update(a[1], r[1])

            if experiment == ['AC', 'LOLA']:
                theta1_ = torch.tensor(agents[0].policy).requires_grad_(True)
                LOLA_loop(agents[1], theta1_, n_lookaheads[1])
                agents[0].update(a[0], r[0])

            for i, exp in enumerate(experiment):
                if exp == 'LOLAom':
                    agents[i].update_logs(np.log(act_probs[i-1]))
                    if update > 1:
                        LOLAom_loop(agents[i], torch.tensor(np.log(act_probs[i-1])), n_lookaheads[i])
                if exp == 'AC':
                    agents[i].update(a[i], r[i])
                if exp == 'ACom':
                    if update > 1:
                        agents[i].set_op_theta(act_probs[i-1])
                        agents[i].update(a[i], r[i], a[i-1])
                if exp == 'AComGP':
                    agents[i].update_logs(act_probs[i-1])
                    if update > 1:
                        AComGP_loop(agents[i], a[i], r[i], a[i-1], act_probs[i-1], n_lookaheads[i])

            a1, a2 = a_s
            r1, r2 = r_s
            if update >= (0.1 * hpL.n_update):
                for rol_a in range(len(a1)):
                    for batch_a in range(len(a1[rol_a])):
                        state_distribution_log[a1[rol_a][batch_a], a2[rol_a][batch_a]] += 1

            ret1, score1 = get_return(r1, u1, mooc)
            ret2, score2 = get_return(r2, u2, mooc)

            if env.NUM_ACTIONS == 2:
                for i in range(len(act_hist_log)):
                    act_hist_log[i].append([update, trial, n_lookaheads[i],
                                            act_probs[i][0], act_probs[i][1]])

                    #trace_log[i].append([update, trial, n_lookaheads[i], ret1[0], ret1[1],
                    #                     act_probs[i][0], act_probs[i][1]])
            else:
                for i in range(len(act_hist_log)):
                    act_hist_log[i].append([update, trial, n_lookaheads[i],
                                            act_probs[i][0], act_probs[i][1], act_probs[i][2]])

                    #trace_log[i].append([update, trial, n_lookaheads[i], ret1[0], ret1[1],
                    #                    act_probs[i][0], act_probs[i][1], act_probs[i][2]])

            payoff_episode_log1.append([update, trial, n_lookaheads[0], score1])
            payoff_episode_log2.append([update, trial, n_lookaheads[1], score2])

        if trial % 5 == 0:
            columns = ['Episode', 'Trial', 'Lookahead', 'Payoff']
            df1 = pd.DataFrame(payoff_episode_log1, columns=columns)
            df2 = pd.DataFrame(payoff_episode_log2, columns=columns)

            path_data = f'results/tour_{experiment}_{game}_l{n_lookaheads[0]}_{n_lookaheads[1]}'  # /{mooc}/{hp.use_baseline}'
            mkdir_p(path_data)

            df1.to_csv(f'{path_data}/agent1_payoff_{info}.csv', index=False)
            df2.to_csv(f'{path_data}/agent2_payoff_{info}.csv', index=False)

            state_distribution = state_distribution_log / (hpL.batch_size * (0.9 * hpL.n_update) * (trial+1) * 100)
            df = pd.DataFrame(state_distribution)
            print(np.sum(state_distribution))
            df.to_csv(f'{path_data}/states_{info}_{n_lookaheads[0]}_{n_lookaheads[1]}.csv', index=False, header=None)

            if env.NUM_ACTIONS == 3:
                columns = ['Episode', 'Trial', 'Lookahead', 'Action 1', 'Action 2', 'Action 3']
                #columns1 = ['Episode', 'Trial', 'Lookahead', 'O1', 'O2', 'Action 1', 'Action 2', 'Action 3']
            else:
                columns = ['Episode', 'Trial', 'Lookahead', 'Action 1', 'Action 2']
                #columns1 = ['Episode', 'Trial', 'Lookahead', 'O1', 'O2', 'Action 1', 'Action 2']
            df1 = pd.DataFrame(act_hist_log[0], columns=columns)
            df2 = pd.DataFrame(act_hist_log[1], columns=columns)

            df1.to_csv(f'{path_data}/agent1_probs_{info}.csv', index=False)
            df2.to_csv(f'{path_data}/agent2_probs_{info}.csv', index=False)

            #df1 = pd.DataFrame(trace_log[0], columns=columns1)
            #df2 = pd.DataFrame(trace_log[1], columns=columns1)

            #df1.to_csv(f'{path_data}/agent1_traces_{info}.csv', index=False)
            #df2.to_csv(f'{path_data}/agent2_traces_{info}.csv', index=False)
            del df1, df2, df
    columns = ['Episode', 'Trial', 'Lookahead', 'Payoff']
    df1 = pd.DataFrame(payoff_episode_log1, columns=columns)
    df2 = pd.DataFrame(payoff_episode_log2, columns=columns)

    path_data = f'results/tour_{experiment}_{game}_l{n_lookaheads[0]}_{n_lookaheads[1]}'  # /{mooc}/{hp.use_baseline}'
    mkdir_p(path_data)

    df1.to_csv(f'{path_data}/agent1_payoff_{info}.csv', index=False)
    df2.to_csv(f'{path_data}/agent2_payoff_{info}.csv', index=False)

    state_distribution = state_distribution_log / (hpL.batch_size * (0.9 * hpL.n_update) * trials * 100)
    df = pd.DataFrame(state_distribution)
    print(np.sum(state_distribution))
    df.to_csv(f'{path_data}/states_{info}_{n_lookaheads[0]}_{n_lookaheads[1]}.csv', index=False, header=None)

    if env.NUM_ACTIONS == 3:
        columns = ['Episode', 'Trial', 'Lookahead', 'Action 1', 'Action 2', 'Action 3']
        # columns1 = ['Episode', 'Trial', 'Lookahead', 'O1', 'O2', 'Action 1', 'Action 2', 'Action 3']
    else:
        columns = ['Episode', 'Trial', 'Lookahead', 'Action 1', 'Action 2']
        # columns1 = ['Episode', 'Trial', 'Lookahead', 'O1', 'O2', 'Action 1', 'Action 2']
    df1 = pd.DataFrame(act_hist_log[0], columns=columns)
    df2 = pd.DataFrame(act_hist_log[1], columns=columns)

    df1.to_csv(f'{path_data}/agent1_probs_{info}.csv', index=False)
    df2.to_csv(f'{path_data}/agent2_probs_{info}.csv', index=False)

    # df1 = pd.DataFrame(trace_log[0], columns=columns1)
    # df2 = pd.DataFrame(trace_log[1], columns=columns1)

    # df1.to_csv(f'{path_data}/agent1_traces_{info}.csv', index=False)
    # df2.to_csv(f'{path_data}/agent2_traces_{info}.csv', index=False)
    del df1, df2, df




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

    parser.add_argument('-trials', type=int, default=30, help="number of trials")
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
    parser.add_argument('-experiment', type=str, default='LOLA-LOLA', help="experiment")

    parser.add_argument('-lookahead1', type=int, default=1, help="number of lookaheads for agent 1")
    parser.add_argument('-lookahead2', type=int, default=1, help="number of lookaheads for agent 2")

    args = parser.parse_args()

    u1 = lambda x: torch.sum(torch.pow(x, 2), dim=0)
    u2 = lambda x: torch.prod(x, dim=0)

    u = [u1, u2]

    n_lookaheads = [args.lookahead1, args.lookahead2]
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


    # for i in range(n_lookaheads):
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    play(n_lookaheads, trials, info, mooc, game, experiment)
