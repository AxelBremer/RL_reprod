from sumtree import SumTree
import numpy as np
import random
import torch

# implementation from https://github.com/rlcode/per/blob/master/SumTree.py

class PrioritizedER():

    # to ensure we do operations on non-zero values
    e = 10e-3

    def __init__(self, capacity, n_episodes, alpha=0.6, beta=0.4):
        self.alpha = alpha
        self.beta = beta
        self.capacity = capacity
        # self.alpha_increment_per_sampling = (1-alpha) / n_episodes
        self.beta_increment_per_sampling = (1-beta) / n_episodes
        # self.beta_increment_per_sampling = 0.001
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

        # clip the hyperparameters to 1
        # self.alpha = np.min(1., self.alpha + self.alpha_increment_per_sampling)
        # self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        if self.tree.n_entries == 0:
            print('JOE JOE: n_entries zijn nul -----------:', self.tree.n_entries)
        elif 0 in sampling_probabilities:
            print('JOE JOE: sampling probabilities zijn nul -----------:', sampling_probabilities)
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        delta = self._get_priority(error)
        self.tree.update(idx, delta)

    def __len__(self):
        return self.tree.n_entries

    def anneal_hyperparams(self):
        # clip the hyperparameters to 1, just in case
        # self.alpha = np.min([1., self.alpha + self.alpha_increment_per_sampling])
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])