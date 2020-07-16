import numpy as np

iga_o1 = np.array([[4, 3, 2],
                   [3, 2, 1],
                   [2, 1, 0]])
iga_o2 = np.array([[0, 1, 2],
                   [1, 2, 3],
                   [2, 3, 4]])


igaNE_o1 = np.array([[4, 1, 2],
                     [3, 3, 1],
                     [1, 2, 1]])
igaNE_o2 = np.array([[1, 2, 1],
                     [1, 2, 2],
                     [2, 1, 3]])


def get_payoff_matrix(game):
    if game == 'iga':
        return [iga_o1, iga_o2]
    elif game == 'igaNE':
       return [igaNE_o1, igaNE_o2]