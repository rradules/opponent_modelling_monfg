import gpytorch
import random
import numpy as np
import torch
from scipy.optimize import minimize


class QLearningAgent:
    def __init__(self, id, hp, utility_function, num_actions):

        self.alpha = hp.alpha
        self.epsilon = hp.epsilon
        self.utility = utility_function
        self.rand_prob = hp.rand_prob
        self.num_objectives = 2
        self.id = id
        self.hp = hp
        self.theta = None

        self.Q = np.zeros((num_actions, 2))
        self.num_actions = num_actions

    # epsilon greedy based on nonlinear optimiser mixed strategy search
    def act(self, state=None, theta=None):
        if random.uniform(0.0, 1.0) < self.epsilon:
            action = self.select_random_action()
        else:
            action = self.select_action_greedy_mixed_nonlinear()
        return np.array([action]), theta

    def select_random_action(self):
        random_action = np.random.randint(self.num_actions)
        return random_action

    # greedy action selection based on nonlinear optimiser mixed strategy search
    def select_action_greedy_mixed_nonlinear(self):
        strategy = self.calc_mixed_strategy_nonlinear()
        if np.sum(strategy) != 1:
                strategy = strategy / np.sum(strategy)
        return np.random.choice(range(self.num_actions), p=strategy)

    def calc_mixed_strategy_nonlinear(self):
        if self.rand_prob:
            s0 = np.random.random(self.num_actions)
            s0 /= np.sum(s0)
        else:
            s0 = np.full(self.num_actions,
                         1.0 / self.num_actions)  # initial guess set to equal prob over all actions

        b = (0.0, 1.0)
        bnds = (b,) * self.num_actions
        con1 = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        cons = ([con1])
        solution = minimize(self.objective, s0, bounds=bnds, constraints=cons)
        strategy = solution.x

        return strategy

    # this is the objective function to be minimised by the nonlinear optimiser
    # Calculates the SER for a given strategy using the agent's own Q values
    # (it returns the negative of SER)
    def objective(self, strategy):
        expected_vec = np.zeros(self.num_objectives)
        for o in range(self.num_objectives):
            expected_vec[o] = np.dot(self.Q[:, o], np.array(strategy))
        ser = self.utility(torch.from_numpy(expected_vec)).item()
        return - ser

    def _apply_discount(self, rewards):
        cum_discount = np.cumprod(self.hp.gamma * np.ones(rewards.shape), axis=0) / self.hp.gamma
        discounted_rewards = np.sum(rewards * cum_discount, axis=0)
        return discounted_rewards

    def update(self, actions, payoff, opp_actions=None):
        # update Q

        actions = np.array(actions)
        # average return over rollout;
        means = self._apply_discount(np.array(payoff))

        for i, act in enumerate(actions[0, :]):
            self.Q[act] += self.alpha * (means[:, i] - self.Q[act])

    def reset(self):
        self.Q = np.zeros((self.num_actions, 2))
        self.epsilon = self.hp.epsilon
