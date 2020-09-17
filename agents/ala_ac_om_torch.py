import numpy as np
from utils.utils import softmax, softmax_grad
from collections import Counter
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCriticAgent:
    def __init__(self, id, hp, utility_function, num_actions):

        self.lr_q = hp.lr_q
        self.lr_theta = hp.lr_theta
        self.utility = utility_function
        self.id = id
        self.hp = hp

        self.Q = nn.Parameter(torch.zeros((num_actions, 2), requires_grad=False))
        self.policy = torch.softmax(nn.Parameter(torch.zeros(num_actions, requires_grad=True)))
        self.theta_optimizer = torch.optim.Adam((self.policy,), lr=hp.lr_out)

        self.num_actions = num_actions

    def act(self, batch_states):

        batch_states = torch.from_numpy(batch_states).long()
        probs = self.policy
        m = Categorical(probs)
        actions = m.sample(sample_shape=batch_states.size())
        log_probs_actions = m.log_prob(actions)

        return actions.numpy().astype(int), log_probs_actions

    def update(self, actions, payoff):
        # update Q
        # average return over rollout; TODO: gamma
        discounted_rewards = self._apply_discount(payoff)

        for i, act in enumerate(actions[0]): #each first action for batch
            self.Q[act] += self.lr_q * (discounted_rewards[:, i] - self.Q[act])

        # expected utility
        u = self.policy @ self.Q

        # gradient: dJ / du
        if self.id == 0:
            grad_u = np.array([2 * u[0], 2 * u[1]])
        else:
            grad_u = np.array([u[1], u[0]])

        grad_pg = softmax_grad(self.policy).T @ self.Q

        grad = grad_u @ grad_pg.T

        # update theta
        self.theta += self.lr_theta * grad

        # update policy
        self.policy = softmax(self.theta)

    def _apply_discount(self, rewards):
        cum_discount = torch.cumprod(self.hp.gamma * torch.ones(*rewards[0].size()), dim=1) / self.hp.gamma
        discounted_rewards = rewards * cum_discount

        return discounted_rewards

    def reset(self):
        self.theta = np.zeros(self.num_actions)
        self.policy = softmax(self.theta)
        self.Q = np.zeros((self.num_actions, 2))


class OppoModelingACAgent:
    def __init__(self, alpha, alpha_theta, utility_function, window, num_actions, utility_id):
        self.alpha = alpha
        self.alpha = 1
        self.alpha_theta = alpha_theta
        self.utility = utility_function
        self.utility_id = utility_id
        self.window = window
        self.num_actions = num_actions

        self.Q = np.zeros((num_actions, num_actions, 2))
        self.theta = np.zeros(num_actions)
        self.policy = softmax(self.theta)

        self.last_action = None
        self.oppo_actions = [i for i in range(num_actions)]

    def sample_action(self):
        action = np.random.choice(range(self.num_actions), p=self.policy)
        self.last_action = action
        return action

    def play(self, payoff_matrix, action1, action2, oppo_action):

        payoff = payoff_matrix[action1][action2]

        # print(self.last_action, oppo_action)
        # print("Payoff: ", payoff)

        # update oppo's action history
        if len(self.oppo_actions) >= self.window:
            self.oppo_actions.pop(0)
            self.oppo_actions.append(oppo_action)
        else:
            self.oppo_actions.append(oppo_action)

        # print("Op actions: ", self.oppo_actions)

        # calculate oppo's action distribution
        oppo_prob = self.count_oppo_actions()

        # print("Opponent strategy: ", oppo_prob)

        # update Q based on oppo's last action
        self.Q[oppo_action][self.last_action] += self.alpha * (payoff - self.Q[oppo_action][self.last_action])
        expected_Q = oppo_prob @ self.Q

        '''
        print("Q-function: ", self.Q)
        print("Marginalized Q-function: ", expected_Q)
        print("Agent policy: ", self.policy)
        '''

        # expected utility based on all oppo's action
        u = self.policy @ expected_Q

        # print("Expected value per objective: ", u)

        # gradient: dJ / du
        if self.utility_id == 1:
            grad_u = np.array([2 * u[0], 2 * u[1]])
        elif self.utility_id == 2:
            grad_u = np.array([u[1], u[0]])
        else:
            grad_u = np.array([-2 * u[0], -2 * u[1]])

        grad_pg = softmax_grad(self.policy).T @ expected_Q
        grad = grad_u @ grad_pg.T
        # update theta
        self.theta += self.alpha_theta * grad

        # update policy
        self.policy = softmax(self.theta)
        return oppo_prob

    def count_oppo_actions(self):
        count = Counter(self.oppo_actions)
        probs = [count[i] / sum(count.values()) for i in range(self.num_actions)]

        return np.array(probs)

    def reset(self):
        self.theta = np.zeros(self.num_actions)
        self.policy = softmax(self.theta)
        self.Q = np.zeros((self.num_actions, self.num_actions, 2))
        self.oppo_actions = [i for i in range(self.num_actions)]

