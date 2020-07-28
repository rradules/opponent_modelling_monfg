import numpy as np

mp1 = np.array([[1, -1],
                [-1, 1]])
mp2 = np.array([[-1, 1],
                [1, -1]])
sg1 = np.array([[0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]])
sg2 = np.array([[0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]])

def get_payoff_matrix(game):
    if game == 'MP':
        return [mp1, mp2]
    elif game == 'SG':
        return [sg1, sg2]
