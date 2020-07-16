"""
(Im)balancing Act Game environment with pure NE under SER.
"""

import numpy as np
from envs.imbal_game_act import ImbalancingActGame


class ImbalancingActGameNE(ImbalancingActGame):
    """
    A two-agent vectorized multi-objective environment with pure NEs under SER
    Possible actions for each agent are (L)eft, (M)iddle and (R)ight.
    """

    def __init__(self, max_steps, batch_size=1):
        super().__init__(max_steps, batch_size)

        self.payoffsObj1 = np.array([[4, 1, 2],
                                     [3, 3, 1],
                                     [1, 2, 1]])
        self.payoffsObj2 = np.array([[1, 2, 1],
                                     [1, 2, 2],
                                     [2, 1, 3]])
        self.payout_mat = [self.payoffsObj1, self.payoffsObj2]

