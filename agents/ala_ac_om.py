import numpy as np
from utils.utils import softmax, softmax_grad
import torch
import gpytorch
from utils.umodel import GradUGPModel
import torch.nn as nn


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

    def update(self, actions, payoff, opp_actions=None):
        # update Q

        actions = np.array(actions)
        # average return over rollout;
        means = self._apply_discount(np.array(payoff))

        for i, act in enumerate(actions[0, :]):
            self.Q[act] += self.lr_q * (means[:, i] - self.Q[act])

        #V = np.max(self.Q, axis=0) A2C?
        # expected return
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

        self.op_theta = []

    def set_op_theta(self, op_theta):
        self.op_theta = op_theta

    def update(self, actions, payoff, opp_actions=None):

        actions = np.array(actions)
        opp_actions = np.array(opp_actions)

        means = self._apply_discount(np.array(payoff))

        for i, act in enumerate(actions[0, :]):  # each first action in rollout
            self.Q[opp_actions[0, i], act, :] += self.lr_q * (means[:, i] - self.Q[opp_actions[0, i], act, :])

        expected_q = self.op_theta @ self.Q

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


class UMOMACAgent(OppoModelingACAgent):
    def __init__(self, id, hp, utility_function, num_actions, hpGP):
        super().__init__(id, hp, utility_function, num_actions)
        self.hpGP = hpGP
        self.theta_log = []
        self.op_theta_log = []

        self.Q = np.zeros((num_actions, num_actions, 2))
        self.op_theta = []

    def set_op_theta(self, op_theta):
        self.op_theta = op_theta

    def update_logs(self, op_theta):
        self.theta_log.append(self.theta)
        self.op_theta_log.append(op_theta)
        self.theta_log = self.theta_log[-self.hpGP.GP_win:]
        self.op_theta_log = self.op_theta_log[-self.hpGP.GP_win:]

    def makeUModel(self):
        y_train = torch.tensor(np.diff(self.op_theta_log, axis=0)/self.hp.lr_theta).float().contiguous()
        op_thetas = torch.tensor(self.op_theta_log)
        thetas = torch.tensor(self.theta_log)

        op_thetas = op_thetas[:-1]
        thetas = thetas[:-1]
        x_train = torch.cat((op_thetas, thetas), dim=1).float().contiguous()

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.num_actions,
                                                                      rank=1)
        umodel = GradUGPModel(self.num_actions, x_train, y_train, likelihood)
        likelihood.train()
        umodel.train()
        optimizer = torch.optim.Adam([
            {'params': umodel.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=self.hpGP.lr_GP)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, umodel)

        for i in range(self.hpGP.iter):
            optimizer.zero_grad()
            output = umodel(x_train)
            loss = -mll(output, y_train)
            loss.backward(retain_graph=True)
            optimizer.step()

        return umodel, likelihood

    def in_lookahead(self, umodel, likelihood):
        umodel.eval()
        likelihood.eval()

        pred_point = torch.cat((torch.tensor(self.op_theta), torch.tensor(self.theta))).float().contiguous()
        pred_point = torch.unsqueeze(pred_point, dim=0)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # TODO add paper discussion about taking mean versus sample GP
            grad = likelihood(umodel(pred_point)).mean

        self.op_theta = self.op_theta + self.hp.lr_theta * grad.numpy()[0]

