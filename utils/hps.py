class HpLolaDice:
    def __init__(self, lr_out=0.2, lr_in=0.3, lr_v=0.1, gamma=0.96, updates=500, rollout=150, batch=128, baseline=False):
        self.lr_out = lr_out
        self.lr_in = lr_in
        self.lr_v = lr_v
        self.gamma = gamma
        self.n_update = updates
        self.len_rollout = rollout
        self.batch_size = batch
        self.use_baseline = baseline


class HpAC:
    def __init__(self, lr_q=0.05, lr_theta=0.05, gamma=0.96, updates=500, rollout=150, batch=128):
        self.lr_q = lr_q
        self.lr_theta = lr_theta
        self.gamma = gamma
        self.n_update = updates
        self.len_rollout = rollout
        self.batch_size = batch
        self.window = 100


class HpPGA_APP:
    def __init__(self):
        #TODO: figure out param values
        self.theta = 0.5
        self.eta = 0.001
        self.xi = 0
        self.gamma = 1  #not RL gamma, more like a lookahead
        self.n_update = 1000
        self.len_rollout = 50
        self.batch_size = 64
        self.epsilon = 0.05

    def update_lr(self, timestep):
        self.eta = 5.0 / (5000 + timestep)
        self.theta = 5.0 / (10 + timestep)


class HpPGA_APP_test:
    def __init__(self):
        self.theta = 0.8
        self.eta = 5.0 / 5000
        self.xi = 0
        self.gamma = 3  #not RL gamma, more like a lookahead
        self.n_update = 20000
        self.len_rollout = 1
        self.epsilon = 0.05

    def update_lr(self, timestep):
        self.eta = 5.0 / (5000 + timestep)
