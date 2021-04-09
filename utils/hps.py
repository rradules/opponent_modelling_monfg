class HpLolaDice:
    def __init__(self, lr_out=0.2, lr_in=0.3, gamma=0.96, updates=500, rollout=150,
                 batch=128, lr_GP=0.1, GP_win=50):
        self.lr_out = lr_out
        self.lr_in = lr_in
        self.gamma = gamma
        self.n_update = updates
        self.len_rollout = rollout
        self.batch_size = batch
        self.lr_GP = lr_GP
        self.GP_win = GP_win


class HpGP:
    def __init__(self, iter=10, lr_GP=0.1, GP_win=50):
        self.iter = iter
        self.lr_GP = lr_GP
        self.GP_win = GP_win


class HpAC:
    def __init__(self, lr_q=0.01, lr_theta=0.1, gamma=0.96, updates=500, rollout=150, batch=128):
        self.lr_q = lr_q
        self.lr_theta = lr_theta
        self.gamma = gamma
        self.n_update = updates
        self.len_rollout = rollout
        self.batch_size = batch
        self.window = 100

class HpQ:
    def __init__(self, alpha=0.05, epsilon=0.1, gamma=0.96, rand_prob=False):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.rand_prob = rand_prob
