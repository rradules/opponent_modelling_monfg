import numpy as np
from utils.utils import softmax, softmax_grad


class ActorCriticAgent:
    def __init__(self, id, hp, utility_function, num_actions):

        self.lr_q = hp.lr_q
        self.lr_theta = hp.lr_theta
        self.utility = utility_function
        self.id = id
        self.hp = hp

        self.Q = np.zeros((num_actions, 2))
        self.theta = np.zeros(num_actions)
        self.policy = softmax(self.theta)
        self.num_actions = num_actions

    def act(self, state=None, theta=None):
        action = np.random.choice(range(self.num_actions), size=self.hp.batch_size,  p=self.policy)
        return action, theta

    def _apply_discount(self, rewards):
        cum_discount = np.cumprod(self.hp.gamma * np.ones(rewards.shape), axis=0) / self.hp.gamma
        discounted_rewards = np.sum(rewards * cum_discount, axis=0)
        return discounted_rewards

    def update(self, actions, payoff, opp_theta=None, opp_actions=None):
        # update Q
        # average return over rollout;
        means = self._apply_discount(np.array(payoff))
        for i, act in enumerate(actions[0]): #each first action for batch
            self.Q[act] += self.lr_q * (means[:, i] - self.Q[act])

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

    def reset(self):
        self.theta = np.zeros(self.num_actions)
        self.policy = softmax(self.theta)
        self.Q = np.zeros((self.num_actions, 2))


class OppoModelingACAgent(ActorCriticAgent):
    def __init__(self, id, hp, utility_function, num_actions):
        super().__init__(id, hp, utility_function, num_actions)

        self.Q = np.zeros((num_actions, num_actions, 2))

    def update(self, actions, payoff, opp_theta=None, opp_actions=None):

        means = self._apply_discount(np.array(payoff))
        for i, act in enumerate(actions[0]):  # each first action for batch
            self.Q[opp_actions[0][i], act, :] += self.lr_q * (means[:, i] - self.Q[opp_actions[0][i], act, :])

        expected_q = opp_theta @ self.Q

        '''
        print("Q-function: ", self.Q)
        print("Marginalized Q-function: ", expected_Q)
        print("Agent policy: ", self.policy)
        '''

        # expected utility based on all oppo's action
        u = self.policy @ expected_q

        # print("Expected value per objective: ", u)

        # gradient: dJ / du
        if self.id == 0:
            grad_u = np.array([2 * u[0], 2 * u[1]])
        else:
            grad_u = np.array([u[1], u[0]])

        grad_pg = softmax_grad(self.policy).T @ expected_q
        grad = grad_u @ grad_pg.T
        # update theta
        self.theta += self.lr_theta * grad

        # update policy
        self.policy = softmax(self.theta)
