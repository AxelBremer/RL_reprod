import os
import numpy as np
import matplotlib.pyplot as plt
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm

import gym

import random
import time
from collections import defaultdict

class QNetwork(nn.Module):
    
    def __init__(self, input_dim, num_hidden, output_dim):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(input_dim, num_hidden)
        self.l2 = nn.Linear(num_hidden, output_dim)

    def forward(self, x):
        relu_output = F.relu(self.l1(x))
        out = self.l2(relu_output)
        return out

class ExperienceReplay:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if len(self.memory) == self.capacity:
            self.memory = self.memory[1:] + [transition]
        else:
            self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def get_epsilon(it):
    return max(1 - it*(0.95/1000),0.05)

def select_action(model, state, epsilon, device):
    if random.random() > epsilon:
        with torch.no_grad():
            return model(torch.tensor([state]).float().to(device)).argmax(dim=1).item()
    else:
        return random.sample([0,1],1)[0]

def get_env(arg):
    if arg == 'M':
        env = gym.envs.make("MountainCar-v0")
    if arg == 'B':
        env = gym.envs.make("BipedalWalker-v2")
    if arg == 'A':
        env = gym.envs.make("Acrobot-v1")
    if arg == 'C':
        env = gym.envs.make("CarRacing-v0")
    return env, env.observation_space, env.action_space

def get_memory(arg, capacity):
    if arg == 'S':
        return ExperienceReplay(capacity)