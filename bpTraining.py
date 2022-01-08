from typing import Deque
import gym
import math
import random
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from bpNet import bpQNet, bpPNet


# setting environment
env = gym.make('BipedalWalker-v3')
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
num_observations = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]


# device
#if torch.cuda.is_available():
if False:   #TODO
    print("running on GPU")
    device = torch.device("cuda")
else:
    print("running on CPU")
    device = torch.device("cpu")


# network
qnet = bpQNet(num_observations, num_actions)
qnet.load("networks/bpQNet")
qnet.to(device)
qnet = qnet.float()

pnet = bpPNet(num_observations, num_actions)
pnet.load("networks/bpPNet")
pnet.to(device)
pnet = pnet.float()


# hyper parameters
EPOCHS = 300
LEARNING_RATE = 0.005
BATCH_SIZE = 30
MEMORY = 10000
MEMORY_RENEWAL = int(1/4 * MEMORY)
DISCOUNT = 0.93
criterion = nn.SmoothL1Loss()
q_optim = optim.SGD(qnet.parameters(), lr=LEARNING_RATE, momentum=0.9)
p_optim = optim.SGD(pnet.parameters(), lr=LEARNING_RATE, momentum=0.9)


class Memory:
    # saves state-action-reward (star) as a Nx. matrix: first n_obs columns (obs), next n_act columns (act), last column (rew)
    # state = observations
    # transition = state + action

    def __init__(self, num_observations, num_actions, capacity):
        self.n_obs = num_observations
        self.n_act = num_actions
        self.star = torch.zeros((capacity, self.n_obs + self.n_act + 1))
        self.length = 0
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.star[index]
    
    def get_stars(self):
        return self.star[:self.length]

    def get_states(self):
        return self.star[:self.length, :self.n_obs]
    
    def get_actions(self):
        return self.star[:self.length, self.n_obs:self.n_obs+self.n_act]
    
    def get_rewards(self):
        return self.star[:self.length, -1]

    def add_state(self, state, action, expected_reward=0):
        self.star[self.length][:self.n_obs] = torch.FloatTensor(state)
        self.star[self.length][self.n_obs:self.n_obs+self.n_act] = torch.FloatTensor(action)
        self.star[self.length][-1] = expected_reward
        self.length += 1

    def add_star(self, star):
        self.star[self.length] = torch.FloatTensor(star)
        self.length += 1
    
    def remove_first_states(self, n):
        np.roll(self.star, -n, axis=0)
        self.star[-n:] = 0
        self.length -= n

    def sample_st_a_r(self, batch_size):
        # yield states, actions and rewards per batch as seperate matrices

        star_copy = torch.clone(self.get_stars())

        #shuffle the tensor
        idx = torch.randperm(self.length)
        star_copy = star_copy[idx].view(star_copy.size())

        for k in range(0, self.length, batch_size):
            batch = star_copy[k:k+batch_size]
            yield (batch[:, :self.n_obs], batch[:, self.n_obs:self.n_obs+self.n_act], batch[:,-1])
        
    def sample_tr_r(self, batch_size):
        # yield transitions (st+a) and rewards per batch as seperate matrices
        
        star_copy = torch.clone(self.get_stars())

        #shuffle the tensor
        idx = torch.randperm(self.length)
        star_copy = star_copy[idx].view(star_copy.size())

        for k in range(0, self.length, batch_size):
            batch = star_copy[k:k+batch_size]
            yield (batch[:, :self.n_obs+self.n_act], batch[:,-1])


memory = Memory(num_observations, num_actions, 2*MEMORY)


def run_and_save_episode(policy):
    global memory

    # run the environment with the current neural network and save the results to memory
    # return the duration of the episode
    observation = env.reset()
    stars = [] #list of transitions
    rewards = []
    done = False
    while not done:
        action = policy.select_action(torch.tensor(observation, device=device), sigma=1).cpu().detach()        # change sigma for epoch-dependent action selection
        stars.append(list(observation) + list(action))
        observation, reward, done, _ = env.step(action)
        rewards.append(reward)
    
    assert len(stars) == len(rewards)
    n = len(rewards)
    cumulative_reward = 0
    for i in range(n-1,-1,-1):
        cumulative_reward = rewards[i] + DISCOUNT * cumulative_reward
        stars[i] += [cumulative_reward]
    
    for star in stars:
        memory.add_star(star)

    return n


def train_qmodel(q_func):
    # do one training cycle
    # return the total loss

    running_loss = 0
    for transitions, rewards in memory.sample_tr_r(BATCH_SIZE):
        #print(q_func(rand))

        q_values = q_func(transitions)
        loss = criterion(q_values, rewards.view(-1,1))
        running_loss += loss.item()

        q_optim.zero_grad()
        loss.backward()
        q_optim.step()

    return running_loss


def train_pmodel(policy, q_func):
    # do one training cycle
    # return the total loss
    running_loss = 0
    policy.train()
    for states, _, _ in memory.sample_st_a_r(BATCH_SIZE):
        actions = policy(states)
        transitions = torch.hstack((states, actions))
        q_vector = q_func(transitions)
        loss = -torch.mean(q_vector, dim=0)     # let the cost be the opposite of the q values of the generated transitions, as our goal is to maximize this q-value
        running_loss += loss.item()

        p_optim.zero_grad()
        loss.backward()
        p_optim.step()

    return running_loss


data = []   # tuples of (epoch, duration)
for epoch in range(EPOCHS):
    print(f"starting epoch {epoch+1}/{EPOCHS}")

    if len(memory) >= MEMORY:
        memory.remove_first_states(MEMORY_RENEWAL)

    while len(memory) < MEMORY:
        duration = run_and_save_episode(pnet)
        data.append((epoch, duration))

    for i in range(5):
        ploss = train_pmodel(pnet, qnet)
        print(f"ploss: {ploss}")

    qloss = train_qmodel(qnet)
    print(f"qloss: {qloss}")


epochs, durations = zip(*data)
epochs = np.array(epochs)
durations = np.array(durations)

plt.scatter(epochs, durations, c='r')

epochs_avg_duration = list(range(EPOCHS))
avg_duration = np.zeros(EPOCHS)       # avg_duration[i] = gemiddelde tijd in epoch i
for epoch in range(EPOCHS):
    elements = durations[epochs==epoch]
    if len(elements) == 0:
        avg_duration[epoch] = avg_duration[epoch-1]
    else:
        average_dur = np.mean(elements)     #get those columns for which the first element is epoch
        avg_duration[epoch] = average_dur

# take a rolling mean of avg_duration
D = 10
smooth_duration = np.zeros(EPOCHS)
for i in range(EPOCHS):
    start = max(i-D, 0)
    end = min(i+D,EPOCHS)
    smooth_duration[i] = np.mean(avg_duration[start:end])

plt.plot(smooth_duration, 'b')


plt.show()

qnet.save("networks/bpQNet")
pnet.save("networks/bpPNet")
    
