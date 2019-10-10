from sumtree import SumTree
import numpy as np
import random
import torch

# implementation from https://github.com/rlcode/per/blob/master/SumTree.py

class PrioritizedER():

    # to ensure we do operations on non-zero values
    e = 10e-3
    beta_increment_per_sampling = 0.001 # TODO: kijken of dit beter kan

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.alpha = alpha
        self.beta = beta
        self.capacity = capacity
        self.tree = SumTree(capacity)

    def _get_priority(self, error):
        return (abs(error) + self.e) ** self.alpha

    # and store in tree:
    def push(self, transition, error):
        delta = self._get_priority(error)
        self.tree.add(delta, transition)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        delta = self._get_priority(error)
        self.tree.update(idx, delta)

    def __len__(self):
        return self.tree.n_entries