import numpy as np
import torch
import pandas as pd
import gpytorch
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import itertools
from utility_model import MultitaskGPModel

info = "0M"  # , "1M"]
n_lookaheads = 3
alpha = 0.2
games = 'iag'  # ['iag', 'iagNE']
path_data1 = f'traces/agent1_traces_{info}_2.csv'
path_data2 = f'traces/agent2_traces_{info}_2.csv'
act_num = 2

df1 = pd.read_csv(path_data1)
df1 = df1.loc[df1['Trial'] == 0]
df2 = pd.read_csv(path_data2)
df2 = df2.loc[df2['Trial'] == 0]

values1 = df1[df1.columns[3:5]].to_numpy()
values2 = df2[df1.columns[3:5]].to_numpy()
df1 = df1[df1.columns[5:]]
df2 = df2[df2.columns[5:]]

theta1 = df1.to_numpy()
theta2 = df2.to_numpy()

train_y1 = torch.tensor(np.diff(theta1, axis=0)/alpha).float().contiguous()
train_y2 = torch.tensor(np.diff(theta2, axis=0)/alpha).float().contiguous()

theta1 = theta1[:-1]
theta2 = theta2[:-1]


values1 = values1[:-1]
values2 = values2[:-1]


train_x1 = torch.tensor(np.concatenate((theta1, theta2), axis=1)).float().contiguous()
train_x2 = torch.tensor(np.concatenate((theta2, theta1), axis=1)).float().contiguous()

state_dict1 = torch.load('traces/gp1.pth')

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=act_num, rank=1)
model1 = MultitaskGPModel(train_x1, train_y1, likelihood)  # Create a new GP model

model1.load_state_dict(state_dict1)


# Get into evaluation (predictive posterior) mode
model1.eval()
likelihood.eval()

indices = np.linspace(0, 0.5, 50)
test_x1 = np.array(list(itertools.product(indices, repeat=2)))
th1 = np.atleast_2d(1 - test_x1[:, 0]).T
th2 = np.atleast_2d(1 - test_x1[:, 1]).T
test_x1 = np.hstack((test_x1[:, [0]], th1, test_x1[:, [1]], th2))
test_x1 = torch.tensor(test_x1).float().contiguous()


with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred1 = likelihood(model1(test_x1))

with torch.no_grad():
    # Initialize plot
    #f, ax = plt.subplots(1, 1, figsize=(4, 3))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get upper and lower confidence bounds
    lower, upper = observed_pred1.confidence_region()
    # Plot training data as black stars
    ax.plot(test_x1[:, 0], test_x1[:, 2], observed_pred1.mean[:, 0].numpy())
    ax.scatter(train_x1[:, 0], train_x1[:, 2], train_y1[:, 0].numpy(), color='r')
    ax.set_xlabel("Theta1")
    ax.set_ylabel("Theta2")
    # Plot predictive means as blue line
    #ax.plot(values1[random_indices, 0], observed_pred.mean[:, 0].numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    #intervals = np.array([lower[:, 0].numpy(), upper[:, 0].numpy()])
    #ax.plot_surface(values1[random_indices, 0], values1[random_indices, 1], intervals, alpha=0.5)
    #ax.set_ylim([-3, 3])
    ax.legend(['Mean', 'Observed Data'])
    plt.show()
