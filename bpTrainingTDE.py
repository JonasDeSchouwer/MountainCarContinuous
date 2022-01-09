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
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torchvision.transforms as T

from bpNet import bpQNet, bpPNet
from utils import Memory, DEVICE, nthroot
from plotting import plot_durations, plot_rewards


# setting environment
env = gym.make('BipedalWalker-v3')
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
num_observations = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]


# network
qnet = bpQNet(num_observations, num_actions)
#qnet.load("networks/bpQNet2")
qnet.to(DEVICE)
qnet = qnet.float()

target_qnet = bpQNet(num_observations, num_actions)
target_qnet.load_state_dict(qnet.state_dict())      # type: ignore
target_qnet.to(DEVICE)

pnet = bpPNet(num_observations, num_actions)
#pnet.load("networks/bpPNet2")
pnet.to(DEVICE)
pnet = pnet.float()

target_pnet = bpPNet(num_observations, num_actions)
target_pnet.load_state_dict(pnet.state_dict())      # type: ignore
target_pnet.to(DEVICE)

# hyper parameters
EPOCHS = 1000
Q_LEARNING_RATE_INIT = 0.1
Q_LEARNING_RATE_FINAL = 0.003
P_LEARNING_RATE_INIT = 0.1
P_LEARNING_RATE_FINAL = 0.003
P_MILESTONES = [30,31, 200, 500]
DISCOUNT = 0.93
TARGET_UPDATE = 5
criterion = nn.SmoothL1Loss()
q_optim = optim.SGD(qnet.parameters(), lr=Q_LEARNING_RATE_INIT, momentum=0.9, nesterov=True, weight_decay=0.005)
p_optim = optim.SGD(pnet.parameters(), lr=P_LEARNING_RATE_INIT, momentum=0.9, nesterov=True, weight_decay=0.005)
q_scheduler = lr_scheduler.StepLR(q_optim, step_size=5, \
    gamma=nthroot(Q_LEARNING_RATE_FINAL/Q_LEARNING_RATE_INIT, EPOCHS/5))
p_scheduler = lr_scheduler.MultiStepLR(p_optim, milestones=P_MILESTONES, \
    gamma=nthroot(P_LEARNING_RATE_FINAL/P_LEARNING_RATE_INIT, len(P_MILESTONES)))

def sigma_scheduler(epoch):
    M1 = 20
    M2 = 75
    if epoch < M1: return 0
    elif epoch < M2: return (epoch-M1) / (M2-M1)
    else: return 1

# practical hyper parameters
BATCH_SIZE = 30
MEMORY = 10000
MEMORY_RENEWAL = int(1/4 * MEMORY)

memory = Memory(num_observations, num_actions, 2*MEMORY)


MONITOR_DATA = []   # add a tuple (epoch, duration, reward) every time an episode is run
def run_and_save_episode(policy, epoch):
    global memory

    # run the environment with the current neural network and save the results to memory
    # add (epoch, duration, reward) to MONITOR_DATA

    total_reward = 0
    duration = 1

    observation = env.reset()
    done = False
    while not done:
        action = policy.select_action(torch.as_tensor(observation, device=DEVICE), sigma=sigma_scheduler(epoch)).cpu().detach()        # change sigma for epoch-dependent action selection
        star = torch.cat((torch.as_tensor(observation, device=DEVICE), torch.as_tensor(action, device=DEVICE)))
        observation, reward, done, _ = env.step(action)
        star = torch.cat((star, torch.as_tensor(observation, device=DEVICE), torch.as_tensor(reward, device=DEVICE).view(1)))
        memory.add_star(star)
        
        duration += 1
        total_reward += reward

    MONITOR_DATA.append((epoch, duration, float(total_reward)))
    print((epoch, duration, float(total_reward)))


def train_qmodel(q_func, q_target, p_target):
    # do one training cycle
    # return the total loss

    running_loss = 0
    for transitions, next_states, rewards in memory.sample_tr_st_r(BATCH_SIZE):
        q_values = q_func(transitions)

        next_transitions = torch.hstack((next_states, p_target(next_states)))
        targets = rewards + DISCOUNT * q_target(next_transitions)

        loss = criterion(q_values, targets)
        running_loss += loss.item()

        q_optim.zero_grad()
        loss.backward()
        q_optim.step()
        q_scheduler.step()

    return running_loss


def train_pmodel(policy, q_target):
    # do one training cycle
    # return the total loss
    running_loss = 0
    policy.train()
    for states in memory.sample_states(BATCH_SIZE):
        actions = policy(states)
        transitions = torch.hstack((states, actions))
        q_vector = q_target(transitions)
        loss = -torch.mean(q_vector, dim=0)     # let the cost be the opposite of the q values of the generated transitions, as our goal is to maximize this q-value
        running_loss += loss.item()

        p_optim.zero_grad()
        loss.backward()
        p_optim.step()
        p_scheduler.step()

    return running_loss


def main():
    try:
        for epoch in range(EPOCHS):
            print(f"\nstarting epoch {epoch+1}/{EPOCHS}")

            if len(memory) >= MEMORY:
                memory.remove_first_states(MEMORY_RENEWAL)

            while len(memory) < MEMORY:
                run_and_save_episode(pnet, epoch)

            for i in range(5):
                ploss = train_pmodel(pnet, qnet)
                print(f"ploss: {ploss}")

            qloss = train_qmodel(qnet, target_qnet, target_pnet)
            print(f"qloss: {qloss}")

            if epoch % TARGET_UPDATE == 0:
                target_qnet.load_state_dict(qnet.state_dict())      # type: ignore
                target_pnet.load_state_dict(pnet.state_dict())      # type: ignore
    except KeyboardInterrupt:
        qnet.save("networks/bpQNet3")
        pnet.save("networks/bpPNet3")
        with open(f"logs/{time.time()}", 'w') as f:
            f.write(str(MONITOR_DATA))
    
    qnet.save("networks/bpQNet3")
    pnet.save("networks/bpPNet3")
    with open(f"logs/{time.time()}", 'w') as f:
        f.write(str(MONITOR_DATA))
    

if __name__ == "__main__":
    main()
