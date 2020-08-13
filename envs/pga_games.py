"""
Iterated Prisoner's dilemma environment.
"""
import gym
import numpy as np

from gym.spaces import Discrete, Tuple

from .common import OneHot


class PGA_Games(gym.Env):
    """
    A two-agent vectorized environment.
    """
    NUM_AGENTS = 2

    def __init__(self, max_steps, payout_mat):
        self.max_steps = max_steps
        self.payout_mat1 = payout_mat[0]
        self.payout_mat2 = payout_mat[1]
        self. NUM_ACTIONS = len(self.payout_mat1[0])

        self.action_space = Tuple([
            Discrete(self.NUM_ACTIONS) for _ in range(self.NUM_AGENTS)
        ])
        self.available_actions = [
            np.ones(self.NUM_ACTIONS, dtype=int)
            for _ in range(self.NUM_AGENTS)
        ]
        self.step_count = None

    def reset(self):
        self.step_count = 0
        info = [{'available_actions': aa} for aa in self.available_actions]
        return info

    def step(self, action):
        ac0, ac1 = action
        self.step_count += 1

        r0 = self.payout_mat1[ac0, ac1]
        r1 = self.payout_mat2[ac0, ac1]

        reward = [r0, r1]
        done = (self.step_count == self.max_steps)
        info = [{'available_actions': aa} for aa in self.available_actions]
        return reward, done, info
