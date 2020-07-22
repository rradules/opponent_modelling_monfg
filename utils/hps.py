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
        self.eta = 0.01
        self.xi = 0.1
        self.gamma = 1  #not RL gamma, more like a lookahead
        self.n_update = 1000
        self.len_rollout = 50
        self.batch_size = 64