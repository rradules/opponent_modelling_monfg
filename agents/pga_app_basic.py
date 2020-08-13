import torch
from torch.distributions import Categorical
import numpy as np


class PGAAPP:
    def __init__(self, env, hp, init_pi=None):
        self.env = env
        # hyperparameters class
        self.hp = hp
        # the MO optimisation criterion (SER/ESR)
        self.pi = init_pi

        # init values and its optimizer
        self.qvalues = np.zeros(env.NUM_ACTIONS)

    def act(self):
        if np.random.rand() < self.hp.epsilon:
            # set equal probabilities for all actions
            probs = np.ones(self.env.NUM_ACTIONS) * (1.0 / self.env.NUM_ACTIONS)
        else:
            # set probabilities according to the policy
            probs = self.pi
        action = np.random.choice(self.env.NUM_ACTIONS, 1, p=probs)[0]
        return action

    def perform_update(self, action, reward):
        max_qs = np.max(self.qvalues)

        self.qvalues[action] = (1 - self.hp.theta) * self.qvalues[action] + \
                               self.hp.theta * (reward + self.hp.xi * max_qs)

        # print("Qvalues: ", self.qvalues)
        values = np.sum(self.pi * self.qvalues)
        # print("Values: ", values)
        # print("Q-V: ", self.qvalues - values)
        for a in range(self.env.NUM_ACTIONS):
            if self.pi[a] == 1.0:
                delta_hat = self.qvalues[a] - values
            else:
                delta_hat = (self.qvalues[a] - values) / (1 - self.pi[a])

            delta = delta_hat - (self.hp.gamma * np.abs(delta_hat) * self.pi[a])
            # print("Delta: ", delta)
            self.pi[a] = self.pi[a] + self.hp.eta * delta

        # projection to valid strategy space
        self.pi = np.clip(self.pi, a_min=0, a_max=1)

        self.pi /= np.sum(self.pi)
        #print("Pi: ", self.pi)
