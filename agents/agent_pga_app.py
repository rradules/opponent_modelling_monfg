import torch
from torch.distributions import Categorical


class AgentPGAAPPBase:
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

        self.pi = torch.rand(env.NUM_ACTIONS)
        self.pi /= torch.sum(self.pi)
        # init values and its optimizer
        if mooc == 'SER':
            self.qvalues = torch.zeros((env.NUM_OBJECTIVES, env.NUM_ACTIONS))
        elif mooc == 'ESR':
            self.qvalues = torch.zeros(env.NUM_ACTIONS)

    def act(self, batch_states):
        batch_states = torch.from_numpy(batch_states).long()
        probs = self.pi #torch.sigmoid(self.pi)

        m = Categorical(probs)
        actions = m.sample(sample_shape=batch_states.size())
        return actions.numpy().astype(int)

    def perform_update(self, action, _, reward):
        if self.mooc == 'ESR':
            reward = self.utility(reward)

        reward = torch.from_numpy(reward).float()
        if self.mooc == 'ESR':
            max_qs, _ = torch.max(self.qvalues, dim=0)
        else:
            max_qs, _ = torch.max(self.qvalues, dim=1)

        if self.mooc == 'SER':
            for i, a in enumerate(action):
                self.qvalues[:, a] = (1 - self.hp.theta) * self.qvalues[:, a] + \
                                     self.hp.theta * (reward[:, i] + (self.hp.xi * max_qs))
        else:
            for i, a in enumerate(action):
                self.qvalues[a] = (1 - self.hp.theta) * self.qvalues[a] + \
                                     self.hp.theta * (reward[i] + (self.hp.xi * max_qs))

        if self.mooc == 'SER':
            qvalues = self.utility(self.qvalues)
        else:
            qvalues = self.qvalues

        values = torch.sum(self.pi * qvalues)
        delta_hat = torch.zeros(self.env.NUM_ACTIONS)
        delta = torch.zeros(self.env.NUM_ACTIONS)

        for a in range(self.env.NUM_ACTIONS):
            if self.pi[a].numpy() == 1.0:
                delta_hat[a] = qvalues[a] - values
            else:
                delta_hat[a] = (qvalues[a] - values)/(1 - self.pi[a])

        delta -= self.hp.gamma * torch.abs(delta_hat) * self.pi
        self.pi += self.hp.eta * delta
        self.pi /= torch.sum(self.pi)