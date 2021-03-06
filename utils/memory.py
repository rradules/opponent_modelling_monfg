class Memory():
    def __init__(self, hp):
        self.self_logprobs = []
        self.op_logprobs = []
        self.rewards = []
        self.hp = hp

    def add(self, lp, op_lp, r):
        self.self_logprobs.append(lp)
        self.op_logprobs.append(op_lp)
        self.rewards.append(r)

    def get_content(self):
        return self.self_logprobs, self.op_logprobs, self.rewards
