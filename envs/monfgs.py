import numpy as np

iag_o1 = np.array([[4, 3, 2],
                   [3, 2, 1],
                   [2, 1, 0]])
iag_o2 = np.array([[0, 1, 2],
                   [1, 2, 3],
                   [2, 3, 4]])

iagNE_o1 = np.array([[4, 1, 2],
                     [3, 3, 1],
                     [1, 2, 1]])
iagNE_o2 = np.array([[1, 2, 1],
                     [1, 2, 2],
                     [2, 1, 3]])

iagM_o1 = np.array([[4, 2],
                    [2, 0]])
iagM_o2 = np.array([[0, 2],
                    [2, 4]])

iagR_o1 = np.array([[4, 3],
                    [3, 2]])
iagR_o2 = np.array([[0, 1],
                    [1, 2]])

iagRNE_o1 = np.array([[4, 1],
                      [3, 3]])
iagRNE_o2 = np.array([[1, 2],
                      [1, 2]])


def get_payoff_matrix(game):
    if game == 'iag':
        return [iag_o1, iag_o2]
    elif game == 'iagNE':
        return [iagNE_o1, iagNE_o2]
    elif game == 'iagR':
        return [iagR_o1, iagR_o2]
    elif game == 'iagM':
        return [iagM_o1, iagM_o2]
    elif game == 'iagRNE':
        return [iagRNE_o1, iagRNE_o2]
