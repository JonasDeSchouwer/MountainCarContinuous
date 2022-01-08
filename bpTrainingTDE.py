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
from utils import Memory


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
#qnet.load("networks/bpQNet2")
qnet.to(device)
qnet = qnet.float()

target_qnet = bpQNet(num_observations, num_actions)
target_qnet.load_state_dict(qnet.state_dict())      # type: ignore

pnet = bpPNet(num_observations, num_actions)
#pnet.load("networks/bpPNet2")
pnet.to(device)
pnet = pnet.float()

target_pnet = bpPNet(num_observations, num_actions)
target_pnet.load_state_dict(pnet.state_dict())      # type: ignore

# hyper parameters
EPOCHS = 100
Q_LEARNING_RATE = 0.01
P_LEARNING_RATE = 0.1
BATCH_SIZE = 30
MEMORY = 10000
MEMORY_RENEWAL = int(1/4 * MEMORY)
DISCOUNT = 0.93
TARGET_UPDATE = 5
criterion = nn.SmoothL1Loss()
q_optim = optim.SGD(qnet.parameters(), lr=Q_LEARNING_RATE, momentum=0.9)
p_optim = optim.SGD(pnet.parameters(), lr=P_LEARNING_RATE, momentum=0.9)
q_scheduler = lr_scheduler.StepLR(q_optim, step_size=5, gamma=0.96)
p_scheduler = lr_scheduler.MultiStepLR(p_optim, milestones=[30], gamma=0.1)

def sigma_scheduler(epoch):
    M1 = 10
    M2 = EPOCHS/5
    if epoch < M1: return 0
    elif epoch < M2: return (epoch-M1) / (M2-M1)
    else: return 1

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
        action = policy.select_action(torch.tensor(observation, device=device), sigma=sigma_scheduler(epoch)).cpu().detach()        # change sigma for epoch-dependent action selection
        star = torch.cat((torch.as_tensor(observation), torch.as_tensor(action)))
        observation, reward, done, _ = env.step(action)
        star = torch.cat((star, torch.as_tensor(observation), torch.as_tensor(reward).view(1)))
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
        #print(q_func(rand))

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


def plot_data(data):
    """
    :param data: list of tuples [(epoch, duration)]
    """

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


def main():
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

    qnet.save("networks/bpQNet2")
    pnet.save("networks/bpPNet2")
    

if __name__ == "__main__":
    main()
