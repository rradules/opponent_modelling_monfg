class HpLolaDice:
    def __init__(self):
        self.lr_out = 0.2
        self.lr_in = 0.3
        self.lr_v = 0.1
        self.gamma = 0.96
        self.n_update = 1000
        self.len_rollout = 50
        self.batch_size = 64
        self.use_baseline = True


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
        #self.theta = 5.0 / (10 + timestep)


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
