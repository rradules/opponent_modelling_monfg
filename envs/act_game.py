"""
(Im)balancing Act Game environment.
"""
import gym
import numpy as np


class ActGame(gym.Env):
    """
    A two-agent vectorized multi-objective environment.
    Possible actions for each agent are (L)eft, (M)iddle and (R)ight.
    """

    NUM_AGENTS = 2


    NUM_OBJECTIVES = 2

    def __init__(self, max_steps, batch_size, payout_mat):
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.NUM_ACTIONS = len(payout_mat[0])
        # s_0 + all action combinations
        self.NUM_STATES = self.NUM_ACTIONS ** 2 + 1
        self.payout_mat = payout_mat

        self.states = np.reshape(np.array(range(self.NUM_ACTIONS**2)) + 1,
                                 (self.NUM_ACTIONS, self.NUM_ACTIONS))

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

    def render(self, mode='human'):
        pass
