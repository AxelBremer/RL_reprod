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

from gym.spaces import Box, Discrete

import utils

from utils import *

import random
import time
# import gym
from collections import defaultdict

import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def push_transition_and_error(model, memory, transition):
    
    s, a, r, n_s, done = transition
    
    # convert to PyTorch datatypes
    s = torch.tensor(s, dtype=torch.float).to(device)
    a = torch.tensor(a, dtype=torch.int64).to(device)
    r = torch.tensor(r, dtype=torch.float).to(device)
    n_s = torch.tensor(n_s, dtype=torch.float).to(device)
    done = torch.tensor(done, dtype=torch.uint8).to(device)

    with torch.no_grad():
        old_target = model(s)[a]
        target = r + config.discount_factor*(model(n_s).max(dim=-1).values)
        target = target * (1-done).float()

    error = abs(target - old_target)
    memory.push(transition, error)

def train_step(model, memory, optimizer, batch_size, discount_factor):    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    if config.replay_type == 'P':
        transitions, idxs, is_weights = memory.sample(batch_size)
    else:
        transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float).to(device)
    action = torch.tensor(action, dtype=torch.int64).to(device)  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float).to(device)
    reward = torch.tensor(reward, dtype=torch.float).to(device)
    done = torch.tensor(done, dtype=torch.uint8).to(device)  # Boolean
    
    # compute the q value
    q_val = compute_q_val(model, state, action)
    
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_target(model, reward, next_state, done, discount_factor)


    if config.replay_type == 'P':
        errors = torch.abs(q_val - target).data.numpy()
        # update priority
        for i in range(batch_size):
            idx = idxs[i]
            memory.memory.update(idx, errors[i])

        # waarom mean?
        loss = (torch.FloatTensor(is_weights) * F.smooth_l1_loss(q_val, target)).mean()
    else:
        # loss is measured from error between current and newly expected Q values
        loss = F.smooth_l1_loss(q_val, target)


    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item(), q_val, target  # Returns a Python scalar, and releases history (similar to .detach())

def main(config):
    print(config.__dict__)

    env, input_space, output_space, env_name = get_env(config.environment)

    memory, mem_name = get_memory(config.replay_type, config.replay_capacity)

    print('env spaces', input_space, output_space)

    path = create_folders(config, env_name, mem_name)

    # Create runs folder if it doesn't yet exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Save the configuration in txt file.
    with open(path+'/args.txt', 'w') as f:
        json.dump(config.__dict__, f, indent=2)

    if isinstance(input_space, Box):
        input_dim = input_space.shape[0]
    elif isinstance(input_space, Discrete):
        input_dim = input_space.n

    if isinstance(output_space, Box):
        output_cont = True
        output_dim = output_space.shape[0]
    elif isinstance(output_space, Discrete):
        output_cont = False
        output_dim = output_space.n

    print('env dimensions', input_dim, output_dim)

    model = QNetwork(input_dim, config.hidden_dim, output_dim)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), config.learning_rate)
    
    mem = False
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    episode_losses = []
    episode_rewards = []
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
            a = select_action(model, st, eps, device)
            st1, r, done, _ = env.step(a)
            
            if config.render: 
                env.render()
                time.sleep(0.01)
            
            transition = (st, a, r, st1, done)
            if mem_name == 'prioritized_replay':
                push_transition_and_error(model, memory, transition)
            else:
                memory.push(transition)
            
            if len(memory) > config.batch_size:
                mem = True
            
            if mem:
                train_loss, q_val, target = train_step(model, memory, optimizer, config.batch_size, config.discount_factor)
                loss += train_loss
                
            st = st1
        episode_durations.append(ct)
        episode_losses.append(loss)
        episode_rewards.append(r)
        env.close()

        d = {'durations':episode_durations, 'losses':episode_losses, 'rewards':episode_rewards}
        with open(path + '/history.json', 'w') as f:
            json.dump(d, f, indent=2)
    
    return episode_durations, episode_losses, episode_rewards

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--replay_type', type=str, default='S', help='Replay type: [S]tandard, [H]indsight, [P]rioritized')
    parser.add_argument('--replay_capacity', type=int, default=1000, help='Number of moves to save in replay memory')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Number of hidden unit')
    parser.add_argument('--environment', type=str, default='C', help='What environment to use: [M]ountainCar, [A]crobot, [B]ipedalWalker, [G]ridworld')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for Adam')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of episodes to train on')
    parser.add_argument('--render', type=bool, default=False, help='Boolean to render environment or not')
    parser.add_argument('--discount_factor', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')

    config = parser.parse_args()

    # Train the model
    main(config)