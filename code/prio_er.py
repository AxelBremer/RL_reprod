import random
import numpy as np
from sumtree import SumTree

class PrioritizedER:
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (abs(error) + self.e) ** self.a

    def push(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            if data == 0:
                p = priorities[-1]
                data = batch[-1]
                idx = idxs[-1]
                print('WARNING: transition value was 0, replaced it with the previous sampled transition')
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = ( priorities / self.tree.total() ) + 10e-5
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
    
    def __len__(self):
        return self.tree.n_entries