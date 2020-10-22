import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

from utils.memory import Memory
from utils.umodel import GradUGPModel
import gpytorch


def magic_box(x):
    return torch.exp(x - x.detach())


class PGDiceBase:
    def __init__(self, id, env, hp, utility, mooc, other_utility=None):
        # own utility function
        self.id = id
        self.utility = utility
        self.other_utility = other_utility
        self.env = env
        # hyperparameters class
        self.hp = hp
        # the MO optimisation criterion (SER/ESR)
        self.mooc = mooc

        self.theta = nn.Parameter(torch.zeros(env.NUM_ACTIONS, requires_grad=True))
        self.op_theta = nn.Parameter(torch.zeros(env.NUM_ACTIONS, requires_grad=True))
        self.theta_optimizer = torch.optim.Adam((self.theta,), lr=hp.lr_out)

    def theta_update(self, objective):
        self.theta_optimizer.zero_grad()
        objective.backward(retain_graph=True)
        self.theta_optimizer.step()

    def set_op_theta(self, op_theta):
        self.op_theta = op_theta

    def in_lookahead(self,  umodel=None, likelihood=None):
        op_memory = self.perform_rollout(self.op_theta, inner=True)
        op_logprobs, logprobs, op_rewards = op_memory.get_content()

        op_objective = self.dice_objective(self.other_utility, op_logprobs, logprobs, op_rewards)
        grad = torch.autograd.grad(op_objective, self.op_theta, create_graph=True)[0]
        self.op_theta = self.op_theta - self.hp.lr_in * grad

    def out_lookahead(self):
        memory = self.perform_rollout(self.op_theta)
        logprobs, other_logprobs, rewards = memory.get_content()

        # update self theta
        objective = self.dice_objective(self.utility, logprobs, other_logprobs, rewards)
        self.theta_update(objective)

    def act(self, batch_states, theta):
        batch_states = torch.tensor(batch_states)
        probs = torch.sigmoid(theta)
        #print(f'Agent {self.id}: {theta}')

        m = Categorical(probs)
        actions = m.sample(sample_shape=batch_states.size())
        log_probs_actions = m.log_prob(actions)
        return actions.numpy().astype(int), log_probs_actions

    def act_opp(self, batch_states, theta):
        batch_states = torch.tensor(batch_states)
        if torch.sum(theta).item() == 1:
            m = Categorical(theta)
        else:
            probs = torch.sigmoid(theta)
            m = Categorical(probs)
        actions = m.sample(sample_shape=batch_states.size())
        log_probs_actions = m.log_prob(actions)
        return actions.numpy().astype(int), log_probs_actions

    def perform_rollout(self, theta, inner=False):
        memory = Memory(self.hp)
        (s1, s2), _ = self.env.reset()
        for t in range(self.hp.len_rollout):
            a1, lp1 = self.act(s1, self.theta)
            a2, lp2 = self.act_opp(s2, theta)
            if self.id > 0:
                (s2, s1), (r2, r1), _, _ = self.env.step((a2, a1))
            else:
                (s1, s2), (r1, r2), _, _ = self.env.step((a1, a2))

            r1 = torch.Tensor(r1)
            r2 = torch.Tensor(r2)

            if inner:
                memory.add(lp2, lp1, r2)
            else:
                memory.add(lp1, lp2, r1)

        return memory

    def dice_objective(self, utility, logprobs, other_logprobs, rewards):
        self_logprobs = torch.stack(logprobs, dim=1)
        other_logprobs = torch.stack(other_logprobs, dim=1)
        # stochastic nodes involved in rewards dependencies:
        dependencies = torch.cumsum(self_logprobs + other_logprobs, dim=1)

        rewards = torch.stack(rewards, dim=2)
        discounted_rewards = self._apply_discount(rewards)

        #print(f'Agent {self.id}: {discounted_rewards}')

        # dice objective:
        if self.mooc == 'SER':
            dice_objective = utility(torch.mean(torch.sum(magic_box(dependencies) * discounted_rewards, dim=2), dim=1))
        else:
            dice_objective = torch.mean(utility(torch.sum(magic_box(dependencies) * discounted_rewards, dim=2)))
        #print(f'Agent {self.id}: {dice_objective}')
        return -dice_objective

    def _apply_discount(self, rewards):
        cum_discount = torch.cumprod(self.hp.gamma * torch.ones(*rewards[0].size()), dim=1) / self.hp.gamma
        discounted_rewards = rewards * cum_discount

        return discounted_rewards


class PGDice1M(PGDiceBase):
    def __init__(self, num, env, hp, utility, other_utility, mooc):
        self.utility = utility
        self.id = num
        self.other_utility = other_utility
        self.env = env
        self.hp = hp
        self.mooc = mooc

        # init theta and its optimizer
        self.theta = nn.Parameter(torch.zeros([env.NUM_STATES, env.NUM_ACTIONS], requires_grad=True))
        self.theta_optimizer = torch.optim.Adam((self.theta,), lr=hp.lr_out)

    def act(self, batch_states, theta):
        batch_states = torch.tensor(batch_states)
        probs = torch.sigmoid(theta)[batch_states]
        m = Categorical(probs)
        actions = m.sample()
        log_probs_actions = m.log_prob(actions)
        return actions.numpy().astype(int), log_probs_actions


class PGDiceOM(PGDiceBase):
    def __init__(self, id, env, hp, utility, mooc, other_utility=None, hpGP=None):
        super(PGDiceOM, self).__init__(id, env, hp, utility, mooc, other_utility)
        self.theta_log = []
        self.op_theta_log = []
        self.hpGP = hpGP

    def update_logs(self, op_theta):
        self.theta_log.append(torch.sigmoid(self.theta))
        self.op_theta_log.append(op_theta)
        self.theta_log = self.theta_log[-self.hpGP.GP_win:]
        self.op_theta_log = self.op_theta_log[-self.hpGP.GP_win:]

    def makeUModel(self):
        y_train = torch.tensor(np.diff(self.op_theta_log, axis=0)/self.hp.lr_in).float().contiguous()
        op_thetas = torch.tensor(self.op_theta_log)
        thetas = torch.stack(self.theta_log)

        op_thetas = op_thetas[:-1]
        thetas = thetas[:-1]
        x_train = torch.cat((op_thetas, thetas), dim=1).float().contiguous()

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.env.NUM_ACTIONS,
                                                                      rank=1)
        umodel = GradUGPModel(self.env.NUM_ACTIONS, x_train, y_train, likelihood)
        likelihood.train()
        umodel.train()
        optimizer = torch.optim.Adam([
            {'params': umodel.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=self.hpGP.lr_GP)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, umodel)

        training_iterations = 10

        for i in range(training_iterations):
            optimizer.zero_grad()
            output = umodel(x_train)
            loss = -mll(output, y_train)
            loss.backward(retain_graph=True)
            #print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()

        return umodel, likelihood

    def in_lookahead(self, umodel, likelihood):
        umodel.eval()
        likelihood.eval()

        pred_point = torch.cat((self.op_theta, torch.sigmoid(self.theta))).float().contiguous()
        pred_point = torch.unsqueeze(pred_point, dim=0)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # TODO add paper discussion about taking mean versus sample GP
            grad = likelihood(umodel(pred_point)).mean
        self.op_theta = self.op_theta - self.hp.lr_in * grad.numpy()[0]

    def _sampleGP(self):
        #TODO add paper discussion about taking mean versus sample GP
        pass
