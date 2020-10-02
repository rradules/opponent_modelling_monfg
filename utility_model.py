import numpy as np
import torch
import pandas as pd
import gpytorch
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

info = "0M"  # , "1M"]
n_lookaheads = 3
alpha = 0.2
games = 'iag'  # ['iag', 'iagNE']
path_data1 = f'traces/agent1_traces_{info}.csv'
path_data2 = f'traces/agent1_traces_{info}.csv'

df1 = pd.read_csv(path_data1)
df2 = pd.read_csv(path_data2)

values1 = df1[df1.columns[3:5]].to_numpy()
values2 = df2[df1.columns[3:5]].to_numpy()
df1 = df1[df1.columns[5:]]
df2 = df2[df2.columns[5:]]

theta1 = df1.to_numpy()
theta2 = df2.to_numpy()

train_y1 = torch.tensor(-np.diff(theta1, axis=0)/alpha).float().contiguous()
train_y2 = torch.tensor(-np.diff(theta2, axis=0)/alpha).float().contiguous()

theta1 = theta1[1:]
theta2 = theta2[1:]
values1 = values1[1:]
values2 = values2[1:]


train_x1 = torch.tensor(np.concatenate((theta1, theta2), axis=1)).float().contiguous()
train_x2 = torch.tensor(np.concatenate((theta2, theta1), axis=1)).float().contiguous()

#train_x1 = torch.tensor(values1).float().contiguous()

print(train_x1.shape)
print(train_y1.shape)

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=3
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=3, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3, rank=1)
model = MultitaskGPModel(train_x1, train_y1, likelihood)

training_iterations = 50
# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x1)
    loss = -mll(output, train_y1)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
    optimizer.step()


print(model.state_dict())
torch.save(model.state_dict(), 'traces/gp1.pth')
#state_dict = torch.load('traces/gp1.pth')

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

random_indices = np.random.choice(499, size=150, replace=False)
test_x1 = train_x1[random_indices, :]

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x1))

x = train_x1.numpy()
x_embedded = TSNE(n_components=2).fit_transform(x)
print(x_embedded.shape)

with torch.no_grad():
    # Initialize plot
    #f, ax = plt.subplots(1, 1, figsize=(4, 3))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.scatter(x_embedded[:, 0], x_embedded[:, 1], train_y1[:, 0].numpy())
    # Plot predictive means as blue line
    #ax.plot(values1[random_indices, 0], observed_pred.mean[:, 0].numpy(), 'b')
    ax.plot(x_embedded[random_indices, 0], x_embedded[random_indices, 1], observed_pred.mean[:, 0].numpy(), 'r')
    # Shade between the lower and upper confidence bounds
    #intervals = np.array([lower[:, 0].numpy(), upper[:, 0].numpy()])
    #ax.plot_surface(values1[random_indices, 0], values1[random_indices, 1], intervals, alpha=0.5)
    #ax.set_ylim([-3, 3])
    ax.legend(['Mean', 'Observed Data'])
    plt.show()

'''
class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([3]), ard_num_dims=3)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([3]), ard_num_dims=3),
            batch_shape=torch.Size([3], ard_num_dims=3)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
model = BatchIndependentMultitaskGPModel(train_x1, train_y1, likelihood)

training_iterations = 50
# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x1)
    print(output.event_shape, output.batch_shape)
    loss = -mll(output, train_y1)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
    optimizer.step()
# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()




info = "0M"  # , "1M"]
n_lookaheads = 5
experiment = 'NLvsNL'

episodes = 50
batch = 64
moocs = 'SER'  # , 'ESR']
games = 'iag'  # ['iag', 'iagNE']
path_data = f'{experiment}_traces_{info}.csv'

df = pd.read_csv(path_data)

print(df.head())

for ep in range(episodes):
    for b in range(batch):
        df_ep = df.loc[(df['Episode'] == ep) & (df['Batch'] == b)]
        r1o1 = df_ep['R1O1'].values
        print(r1o1.shape)
'''


