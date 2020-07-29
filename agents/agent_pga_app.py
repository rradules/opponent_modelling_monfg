import torch
from torch.distributions import Categorical
import numpy as np


class AgentPGAAPPBase:
    def __init__(self, env, hp, utility, other_utility, mooc, init_pi=None):
        # own utility function
        self.utility = utility
        # opponent utility function
        self.other_utility = other_utility
        self.env = env
        # hyperparameters class
        self.hp = hp
        # the MO optimisation criterion (SER/ESR)
        self.mooc = mooc
        if init_pi:
            self.pi = torch.FloatTensor(init_pi)
        else:
            self.pi = torch.rand(env.NUM_ACTIONS)
            self.pi /= torch.sum(self.pi)

        # init values and its optimizer
        if mooc == 'SER':
            self.qvalues = torch.zeros((env.NUM_OBJECTIVES, env.NUM_ACTIONS))
        elif mooc == 'ESR':
            self.qvalues = torch.zeros(env.NUM_ACTIONS)

    def act(self, batch_states):
        rand = np.random.rand()
        batch_states = torch.from_numpy(batch_states).long()
        if rand < self.hp.epsilon:
            # set equal probabilities for all actions
            probs = torch.from_numpy(np.ones(self.env.NUM_ACTIONS) * (1.0/self.env.NUM_ACTIONS))
        else:
            # set probabilities according to the policy
            probs = self.pi
        m = Categorical(probs)
        actions = m.sample(sample_shape=batch_states.size())
        return actions.numpy().astype(int)

    def perform_update(self, action, _, reward):
        if self.mooc == 'SER':
            max_qs, _ = torch.max(self.qvalues, dim=1)
            reward = torch.from_numpy(reward).float()

            # for each action in the batch
            for i, a in enumerate(action):
                self.qvalues[:, a] = (1 - self.hp.theta) * self.qvalues[:, a] + \
                                     self.hp.theta * (reward[:, i] + (self.hp.xi * max_qs))
            qvalues = self.utility(self.qvalues)

        else:
            reward = self.utility(reward)
            max_qs, _ = torch.max(self.qvalues, dim=0)
            reward = torch.from_numpy(reward).float()

            # for each action in the batch
            for i, a in enumerate(action):
                self.qvalues[a] = (1 - self.hp.theta) * self.qvalues[a] + \
                                     self.hp.theta * (reward[i] + (self.hp.xi * max_qs))
            qvalues = self.qvalues


        #print("Qvalues: ", qvalues)
        values = torch.sum(self.pi * qvalues)
        #print("Values: ", values)
        #print("Q-V: ", qvalues - values)
        for a in range(self.env.NUM_ACTIONS):
            if self.pi[a].numpy() == 1.0:
                delta_hat = qvalues[a] - values
            else:
                delta_hat = (qvalues[a] - values)/(1 - self.pi[a])

            delta = delta_hat - self.hp.gamma * torch.abs(delta_hat) * self.pi[a]
            # print("Delta: ", delta)
            self.pi[a] = self.pi[a] + self.hp.eta * delta

        # projection to valid strategy space
        self.pi = torch.clamp(self.pi, max=1)
        self.pi = torch.clamp(self.pi, min=0)

        self.pi /= torch.sum(self.pi)


class AgentPGAAPP1M(AgentPGAAPPBase):

    def __init__(self, env, hp, utility, other_utility, mooc):
        super().__init__(env, hp, utility, other_utility, mooc)

        self.pi = torch.rand((env.NUM_ACTIONS, env.NUM_STATES))
        self.pi /= torch.sum(self.pi)
        # init q-values
        if mooc == 'SER':
            self.qvalues = torch.zeros((env.NUM_OBJECTIVES, env.NUM_ACTIONS, env.NUM_STATES))
        elif mooc == 'ESR':
            self.qvalues = torch.zeros((env.NUM_ACTIONS, env.NUM_STATES))

    def perform_update(self, action, state, reward):
        if self.mooc == 'SER':
            reward = torch.from_numpy(reward).float()
            max_qs, _ = torch.max(self.qvalues, dim=1)

            for i, (a, s) in enumerate(zip(action, state)):
                self.qvalues[:, a, s] = (1 - self.hp.theta) * self.qvalues[:, a, s] + \
                                     self.hp.theta * (reward[:, i] + (self.hp.xi * max_qs[:, s]))
            qvalues = self.utility(self.qvalues)

        else:
            reward = self.utility(reward)
            max_qs, _ = torch.max(self.qvalues, dim=0)
            reward = torch.from_numpy(reward).float()

            for i, (a, s) in enumerate(zip(action, state)):
                self.qvalues[a, s] = (1 - self.hp.theta) * self.qvalues[a, s] + \
                                     self.hp.theta * (reward[i] + (self.hp.xi * max_qs[s]))
            qvalues = self.qvalues

        values = torch.sum(self.pi * qvalues, dim=0)
        #delta_hat = torch.zeros(self.env.NUM_ACTIONS, self.env.NUM_STATES)
        #delta = torch.zeros(self.env.NUM_ACTIONS, self.env.NUM_STATES)

        for s in state:
            for a in range(self.env.NUM_ACTIONS):
                if self.pi[a, s].numpy() == 1.0:
                    delta_hat = qvalues[a, s] - values[s]
                else:
                    delta_hat = (qvalues[a, s] - values[s])/(1 - self.pi[a, s])

                delta = delta_hat - self.hp.gamma * torch.abs(delta_hat) * self.pi[a, s]
                self.pi[a, s] += self.hp.eta * delta

        # projection to valid strategy space
        self.pi = torch.clamp(self.pi, max=1)
        self.pi = torch.clamp(self.pi, min=0)

        self.pi /= torch.sum(self.pi)

    def act(self, batch_states):
        rand = np.random.rand()
        batch_states = torch.from_numpy(batch_states).long()
        if rand < self.hp.epsilon:
            # set equal probabilities for all actions
            probs = torch.from_numpy(np.ones(self.env.NUM_ACTIONS) * (1.0 / self.env.NUM_ACTIONS))
            probs = probs.repeat(self.hp.batch_size, 1)
        else:
            # set probabilities according to the policy, for every state in the batch
            probs = self.pi[:, batch_states].permute(1, 0)
        m = Categorical(probs)
        actions = m.sample()
        return actions.numpy().astype(int)
