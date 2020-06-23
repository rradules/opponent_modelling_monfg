class Memory():
    def __init__(self, hp):
        self.self_logprobs = []
        self.other_logprobs = []
        self.values = []
        self.rewards = []
        self.hp = hp

    def add(self, lp, other_lp, v, r):
        self.self_logprobs.append(lp)
        self.other_logprobs.append(other_lp)
        self.values.append(v)
        self.rewards.append(r)

    def get_content(self):
        return self.self_logprobs, self.other_logprobs, self.values, self.rewards
