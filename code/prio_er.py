import numpy as np
import random

# implementation from https://github.com/rlcode/per/blob/master/SumTree.py
class SumTree():

    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

##############################################################

class PrioritizedER():

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.e = 10e-3
        self.alpha = alpha
        self.beta = beta
        self.capacity = capacity
        self.beta_increment_per_sampling = 0.001 # TODO: kijken of dit beter kan
        self.memory = SumTree(capacity)

    def _get_priority(self, error):
        # nog door sommatie delen?
        return (np.abs(error) + self.e) ** self.alpha

    def push(self, transition, error):
        # add the td error to the memory sample
        # transition = (transition, error)

        # and store in memory:
        p = self._get_priority(error)
        self.memory.add(p, transition)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.memory.total() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.memory.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.memory.total()
        is_weight = np.power(self.memory.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def __len__(self):
        return self.memory.n_entries