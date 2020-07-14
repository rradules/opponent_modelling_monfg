"""
(Im)balancing Act Game environment.
"""
import gym
import numpy as np
from gym.spaces import Discrete, Tuple

from .common import OneHot


class ImbalancingActGameNE(gym.Env):
    """
    A two-agent vectorized multi-objective environment.
    Possible actions for each agent are (L)eft, (M)iddle and (R)ight.
    """
    NUM_AGENTS = 2
    NUM_ACTIONS = 3
    # s_0 + all action combinations
    NUM_STATES = 10

    def __init__(self, max_steps, batch_size=1):
        self.max_steps = max_steps
        self.batch_size = batch_size

        self.payoffsObj1 = np.array([[4, 1, 2],
                                     [3, 3, 1],
                                     [1, 2, 1]])
        self.payoffsObj2 = np.array([[1, 2, 1],
                                     [1, 2, 2],
                                     [2, 1, 3]])

        self.payout_mat = [self.payoffsObj1, self.payoffsObj2]

        self.states = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

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
        r = np.array([el[ac0, ac1] for el in self.payout_mat])
        s0 = self.states[ac0, ac1]
        s1 = self.states[ac1, ac0]
        observation = [s0, s1]
        reward = [r, r]
        done = (self.step_count == self.max_steps)
        info = [{'available_actions': aa} for aa in self.available_actions]
        return observation, reward, done, info
