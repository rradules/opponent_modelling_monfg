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
        # TODO: ensure enough exploration
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


class AgentPGAAPP1M(AgentPGAAPPBase):
    def __init__(self, env, hp, utility, other_utility, mooc):
        super().__init__(env, hp, utility, other_utility, mooc)

        self.pi = torch.rand((env.NUM_ACTIONS, env.NUM_STATES))
        self.pi /= torch.sum(self.pi)

        # init values and its optimizer
        if mooc == 'SER':
            self.qvalues = torch.zeros((env.NUM_OBJECTIVES, env.NUM_ACTIONS, env.NUM_STATES))
        elif mooc == 'ESR':
            self.qvalues = torch.zeros((env.NUM_ACTIONS, env.NUM_STATES))

    def perform_update(self, action, state, reward):
        if self.mooc == 'ESR':
            reward = self.utility(reward)
        reward = torch.from_numpy(reward).float()

        if self.mooc == 'ESR':
            max_qs, _ = torch.max(self.qvalues, dim=0)
        else:
            max_qs, _ = torch.max(self.qvalues, dim=1)

        if self.mooc == 'SER':
            for i, (a, s) in enumerate(zip(action, state)):
                self.qvalues[:, a, s] = (1 - self.hp.theta) * self.qvalues[:, a, s] + \
                                     self.hp.theta * (reward[:, i] + (self.hp.xi * max_qs[:, s]))
        else:
            for i, (a, s) in enumerate(zip(action, state)):
                self.qvalues[a, s] = (1 - self.hp.theta) * self.qvalues[a, s] + \
                                     self.hp.theta * (reward[i] + (self.hp.xi * max_qs[s]))

        if self.mooc == 'SER':
            qvalues = self.utility(self.qvalues)
        else:
            qvalues = self.qvalues

        values = torch.sum(self.pi * qvalues, dim=0)
        delta_hat = torch.zeros(self.env.NUM_ACTIONS, self.env.NUM_STATES)
        delta = torch.zeros(self.env.NUM_ACTIONS, self.env.NUM_STATES)

        for s in state:
            for a in range(self.env.NUM_ACTIONS):
                if self.pi[a, s].numpy() == 1.0:
                    delta_hat[a, s] = qvalues[a, s] - values[s]
                else:
                    delta_hat[a, s] = (qvalues[a, s] - values[s])/(1 - self.pi[a, s])

        delta -= self.hp.gamma * torch.abs(delta_hat) * self.pi
        self.pi += self.hp.eta * delta
        self.pi /= torch.sum(self.pi)

    def act(self, batch_states):
        # TODO: ensure enough exploration
        batch_states = torch.from_numpy(batch_states).long()
        probs = self.pi[:, batch_states].permute(1, 0)
        m = Categorical(probs)
        actions = m.sample()
        return actions.numpy().astype(int)