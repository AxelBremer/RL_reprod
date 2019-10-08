import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import gym
import json

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm

import utils

from utils import *

import random
import time
# import gym
from collections import defaultdict

import argparse

def compute_q_val(model, state, action):
    output = model(state)
    qs = output[list(range(output.shape[0])),action]
    return qs
    
def compute_target(model, reward, next_state, done, discount_factor):
    return reward + discount_factor*(model(next_state).max(dim=1)).values * (1-done).float()

def train_step(model, memory, optimizer, batch_size, discount_factor):    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)
    done = torch.tensor(done, dtype=torch.uint8)  # Boolean
    
    # compute the q value
    q_val = compute_q_val(model, state, action)
    
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_target(model, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def main(config):
    # Create runs folder if it doesn't yet exist
    if not os.path.exists('runs'):
        os.makedirs('runs')
    # Create folder for this run
    if not os.path.exists(f'runs/{config.name}'):
        os.makedirs(f'runs/{config.name}')

    print(config.__dict__)

    # Save the configuration in txt file.
    with open(f'runs/{config.name}/args.txt', 'w') as f:
        json.dump(config.__dict__, f, indent=2)

    env, input_dim, output_dim = get_env(config.environment)

    print('env dimensions', input_dim, output_dim)

    model = QNetwork(input_dim.shape[0], config.hidden_dim, output_dim.n)

    optimizer = optim.Adam(model.parameters(), config.learning_rate)

    memory = get_memory(config.replay_type, config.replay_capacity)
    
    mem = False
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    episode_losses = []
    for i in _tqdm(range(config.num_episodes)):
        st = env.reset()
        if config.render: env.render()
        
        ct = 0
        loss = 0
        done = False
        
        while not done:
            ct += 1
            global_steps += 1
            
            eps = get_epsilon(global_steps)
            a = select_action(model, st, eps)
            st1, r, done, _ = env.step(a)
            
            if config.render: 
                env.render()
                time.sleep(0.01)
            
            memory.push((st, a, r, st1, done))
            
            if len(memory) > config.batch_size:
                mem = True
            
            if mem:
                loss += train_step(model, memory, optimizer, config.batch_size, config.discount_factor)
                
            st = st1
        episode_durations.append(ct)
        episode_losses.append(loss)
        env.close()
    return episode_durations

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--name', type=str, required=True, help='name of run')
    parser.add_argument('--replay_type', type=str, default='S', help='Replay type: [S]tandard, [H]indsight, [P]rioritized')
    parser.add_argument('--replay_capacity', type=int, default=1000, help='Number of moves to save in replay memory')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Number of hidden unit')
    parser.add_argument('--environment', type=str, default='C', help='What environment to use: [C]artpole')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for Adam')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of episodes to train on')
    parser.add_argument('--render', type=bool, default=False, help='Boolean to render environment or not')
    parser.add_argument('--discount_factor', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')

    config = parser.parse_args()

    # Train the model
    main(config)