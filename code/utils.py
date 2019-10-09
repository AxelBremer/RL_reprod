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
from collections import deque

class QNetwork(nn.Module):
    
    def __init__(self, input_dim, num_hidden, output_dim):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(input_dim, num_hidden)
        self.l2 = nn.Linear(num_hidden, output_dim)

    def forward(self, x):
        relu_output = F.relu(self.l1(x))
        out = self.l2(relu_output)
        return out


class HindsightExperienceReplay:
    def __init__(self):
        self.buffer = deque()

    def reset(self):
        self.buffer = deque()

    def keep(self,item):
        self.buffer.append(item)

    def backward(self):
        num = len(self.buffer)

        # goal = self.buffer[-1][-2][1,:,:]
        last_transition = self.buffer[-1]
        last_reward = last_transition[2]

        # Imagine the last goal to be a goal...
        if last_reward != 0:
            last_transition = list(last_transition)
            last_transition[2] = 0
            self.buffer[-1] = tuple(last_transition)

        return self.buffer


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
        name = 'mountain_car'
    if arg == 'B':
        env = gym.envs.make("BipedalWalker-v2")
        name = 'bipedial'
    if arg == 'A':
        env = gym.envs.make("Acrobot-v1")
        name = 'acrobot'
    return env, env.observation_space, env.action_space, name


def get_memory(arg, capacity):
    if arg == 'S':
        return ExperienceReplay(capacity), 'uniform_replay'
    if arg == 'H':
        return ExperienceReplay(capacity), 'uniform_hindsight_replay'

def create_folders(config, env_name, mem_name):
    # Create runs folder if it doesn't yet exist
    if not os.path.exists('runs'):
        os.makedirs('runs')

    # Create runs folder if it doesn't yet exist
    if not os.path.exists(f'runs/{env_name}'):
        os.makedirs(f'runs/{env_name}')

    # Create runs folder if it doesn't yet exist
    if not os.path.exists(f'runs/{env_name}/{mem_name}'):
        os.makedirs(f'runs/{env_name}/{mem_name}')

    # Create runs folder if it doesn't yet exist
    if not os.path.exists(f'runs/{env_name}/{mem_name}/buffer_{config.replay_capacity}'):
        os.makedirs(f'runs/{env_name}/{mem_name}/buffer_{config.replay_capacity}')

    num = len(os.listdir(f'runs/{env_name}/{mem_name}/buffer_{config.replay_capacity}'))

    path = f'runs/{env_name}/{mem_name}/buffer_{config.replay_capacity}/{num+1}'

    return path