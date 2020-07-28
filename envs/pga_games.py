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
    Possible actions for each agent are (C)ooperate and (D)efect.
    """
    # Possible actions
    NUM_AGENTS = 2

    def __init__(self, max_steps, batch_size, payout_mat):
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.payout_mat1 = payout_mat[0]
        self.payout_mat2 = payout_mat[1]
        self. NUM_ACTIONS = len(self.payout_mat1[0])
        self. NUM_STATES = self.NUM_ACTIONS**2 + 1

        self.states = np.reshape(np.array(range(self.NUM_ACTIONS**2)) + 1,
                                 (self.NUM_ACTIONS, self.NUM_ACTIONS))

        self.action_space = Tuple([
            Discrete(self.NUM_ACTIONS) for _ in range(self.NUM_AGENTS)
        ])
        self.observation_space = Tuple([
            OneHot(self.NUM_STATES) for _ in range(self.NUM_AGENTS)
        ])
        self.available_actions = [
            np.ones((batch_size, self.NUM_ACTIONS), dtype=int)
            for _ in range(self.NUM_AGENTS)
        ]
        self.step_count = None

    def reset(self):
        self.step_count = 0
        init_state = np.zeros(self.batch_size)
        observation = [init_state, init_state]
        info = [{'available_actions': aa} for aa in self.available_actions]
        return observation, info

    def step(self, action):
        ac0, ac1 = action
        self.step_count += 1

        r0 = self.payout_mat1[ac0, ac1]
        r1 = self.payout_mat2[ac0, ac1]
        s0 = self.states[ac0, ac1]
        s1 = self.states[ac1, ac0]
        observation = [s0, s1]
        reward = [[r0, r0], [r1, r1]]
        done = (self.step_count == self.max_steps)
        info = [{'available_actions': aa} for aa in self.available_actions]
        return observation, reward, done, info
