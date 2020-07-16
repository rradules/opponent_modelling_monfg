import torch
import torch.nn as nn
from torch.distributions import Categorical

from utils.memory import Memory


def magic_box(x):
    return torch.exp(x - x.detach())


class AgentDiceBase:
    def __init__(self, env, hp, utility, other_utility, mooc):
        # own utility function
        self.utility = utility
        # opponent utility function
        self.other_utility = other_utility
        self.env = env
        # hyperparameters class
        self.hp = hp
        # the MO optimisation criterion (SER/ESR)
        self.mooc = mooc

        self.theta = nn.Parameter(torch.zeros(env.NUM_ACTIONS, requires_grad=True))
        self.theta_optimizer = torch.optim.Adam((self.theta,), lr=hp.lr_out)
        # init values and its optimizer
        if mooc == 'SER':
            self.values = nn.Parameter(torch.zeros(env.NUM_OBJECTIVES, requires_grad=True))
        elif mooc == 'ESR':
            self.values = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.value_optimizer = torch.optim.Adam((self.values,), lr=hp.lr_v)

    def theta_update(self, objective):
        self.theta_optimizer.zero_grad()
        objective.backward(retain_graph=True)
        self.theta_optimizer.step()

    def value_update(self, loss):
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

    def in_lookahead(self, other_theta, other_values):
        other_memory = self.perform_rollout(other_theta, other_values, inner=True)
        logprobs, other_logprobs, values, rewards = other_memory.get_content()

        other_objective = self.dice_objective(self.other_utility, logprobs, other_logprobs, values, rewards)
        grad = torch.autograd.grad(other_objective, other_theta, create_graph=True)[0]
        return grad

    def out_lookahead(self, other_theta, other_values):
        memory = self.perform_rollout(other_theta, other_values)
        logprobs, other_logprobs, values, rewards = memory.get_content()

        # update self theta
        objective = self.dice_objective(self.utility, logprobs, other_logprobs, values, rewards)
        self.theta_update(objective)
        # update self value:
        v_loss = self.value_loss(values, rewards)
        self.value_update(v_loss)

    def act(self, batch_states, theta, values):
        batch_states = torch.from_numpy(batch_states).long()
        probs = torch.sigmoid(theta)

        m = Categorical(probs)
        actions = m.sample(sample_shape=batch_states.size())
        log_probs_actions = m.log_prob(actions)
        return actions.numpy().astype(int), log_probs_actions, values.repeat(self.hp.batch_size, 1)

    def perform_rollout(self, theta, values, inner=False):
        memory = Memory(self.hp)
        (s1, s2), _ = self.env.reset()
        for t in range(self.hp.len_rollout):
            a1, lp1, v1 = self.act(s1, self.theta, self.values)
            a2, lp2, v2 = self.act(s2, theta, values)
            (s1, s2), (r1, r2), _, _ = self.env.step((a1, a2))
            if self.mooc == 'ESR':
                r1 = self.utility(r1)
                r2 = self.utility(r2)
            if inner:
                memory.add(lp2, lp1, v2, torch.from_numpy(r2).float())
            else:
                memory.add(lp1, lp2, v1, torch.from_numpy(r1).float())

        return memory

    def dice_objective(self, utility, logprobs, other_logprobs, values, rewards):
        self_logprobs = torch.stack(logprobs, dim=1)
        other_logprobs = torch.stack(other_logprobs, dim=1)
        values = torch.stack(values, dim=2).permute(1, 0, 2)
        if self.mooc == 'SER':
            rewards = torch.stack(rewards, dim=2)
            cum_discount = torch.cumprod(self.hp.gamma * torch.ones(*rewards[0].size()), dim=1) / self.hp.gamma
            discounted_rewards, discounted_values = self._apply_discount_SER(cum_discount, rewards, values, utility)
        else:
            rewards = torch.stack(rewards, dim=-1)
            cum_discount = torch.cumprod(self.hp.gamma * torch.ones(*rewards.size()), dim=1) / self.hp.gamma
            discounted_rewards, discounted_values = self._apply_discount_ESR(cum_discount, rewards, values)

        # stochastic nodes involved in rewards dependencies:
        dependencies = torch.cumsum(self_logprobs + other_logprobs, dim=1)

        # logprob of each stochastic nodes:
        stochastic_nodes = self_logprobs + other_logprobs

        # dice objective:
        dice_objective = torch.mean(torch.sum(magic_box(dependencies) * discounted_rewards, dim=1))

        if self.hp.use_baseline:
            # variance_reduction:
            baseline_term = torch.mean(torch.sum((1 - magic_box(stochastic_nodes)) * discounted_values, dim=1))
            dice_objective = dice_objective + baseline_term

        return -dice_objective  # want to minimize -objective

    def value_loss(self, values, rewards):
        if self.mooc == 'SER':
            values = torch.stack(values, dim=1).permute(2, 1, 0)
            rewards = torch.stack(rewards, dim=1)
        else:
            values = torch.stack(values, dim=1)
            rewards = torch.stack(rewards, dim=1).unsqueeze(2)
        return torch.mean((rewards - values) ** 2)

    def _apply_discount_SER(self, cum_discount, rewards, values, utility):
        discounted_rewards_o1 = rewards[0] * cum_discount
        discounted_rewards_o2 = rewards[1] * cum_discount
        discounted_values_1 = values[0] * cum_discount
        discounted_values_o2 = values[1] * cum_discount

        discounted_rewards = utility([discounted_rewards_o1, discounted_rewards_o2])
        discounted_values = utility([discounted_values_1, discounted_values_o2])

        return discounted_rewards, discounted_values

    def _apply_discount_ESR(self, cum_discount, rewards, values):
        discounted_rewards = rewards * cum_discount
        discounted_values = values * cum_discount

        return discounted_rewards, discounted_values


class AgentDice1M(AgentDiceBase):
    def __init__(self, env, hp, utility, other_utility, mooc):
        self.utility = utility
        self.other_utility = other_utility
        self.env = env
        self.hp = hp
        self.mooc = mooc

        # init theta and its optimizer
        self.theta = nn.Parameter(torch.zeros([env.NUM_STATES, env.NUM_ACTIONS], requires_grad=True))
        self.theta_optimizer = torch.optim.Adam((self.theta,), lr=hp.lr_out)
        # init values and its optimizer
        if mooc == 'SER':
            self.values = nn.Parameter(torch.zeros([env.NUM_STATES, env.NUM_OBJECTIVES], requires_grad=True))
        else:
            self.values = nn.Parameter(torch.zeros([env.NUM_STATES, 1], requires_grad=True))
        self.value_optimizer = torch.optim.Adam((self.values,), lr=hp.lr_v)

    def act(self, batch_states, theta, values):
        batch_states = torch.from_numpy(batch_states).long()
        probs = torch.sigmoid(theta)[batch_states]

        m = Categorical(probs)
        actions = m.sample()
        log_probs_actions = m.log_prob(actions)
        return actions.numpy().astype(int), log_probs_actions, values[batch_states]
