from typing import List
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import time
import datetime
import configparser
import os
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torchvision.transforms as T

from mccNet import mccQNet, mccPNet
from utils import Memory, DEVICE, nthroot, soft_update
from plotting import plot_durations, plot_rewards


# configuration
CONFIG = configparser.ConfigParser()
assert len(CONFIG.read(r"configurations\medium_from_scratch.ini"))>0, "config file could not be opened"
print("CONFIGURATION:", CONFIG['general']['name'])


# make task
timestring = datetime.datetime.fromtimestamp(time.time()).strftime('%y.%m.%d-%H.%M')
TASK_DIR = os.path.join("tasks", timestring+'_'+CONFIG['general']['name'])
if os.path.exists(TASK_DIR):
    i = 1
    while os.path.exists(TASK_DIR+str(i)):
        i += 1
    TASK_DIR = TASK_DIR + str(i)
        
os.mkdir(TASK_DIR)
Q_SAVE_LOCATION = os.path.join(TASK_DIR, 'qNet')
Q_BEST_LOCATION = os.path.join(TASK_DIR, 'qBest')
P_SAVE_LOCATION = os.path.join(TASK_DIR, 'pNet')
P_BEST_LOCATION = os.path.join(TASK_DIR, 'pBest')
VIDEO_BASE_PATH = os.path.join(TASK_DIR, 'recordings')
os.mkdir(VIDEO_BASE_PATH)


# logging
LOG_PATH = os.path.join(TASK_DIR, "log.txt")
logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger("")
logger.setLevel(logging.DEBUG)
console_handle = logging.StreamHandler()
console_handle.setLevel(logging.INFO)
file_handle = logging.FileHandler(filename=LOG_PATH)
file_handle.setLevel(logging.DEBUG)
logger.addHandler(console_handle)
logger.addHandler(file_handle)


# setting environment
env = gym.make('MountainCarContinuous-v0')
num_observations = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]


# network
qnet = mccQNet(num_observations, num_actions)
if CONFIG['network']['q_load'] not in ["", "None", "NONE"]:
    qnet.load(CONFIG['network']['q_load'])
qnet.to(DEVICE)
qnet = qnet.float()

target_qnet = mccQNet(num_observations, num_actions)
target_qnet.load_state_dict(qnet.state_dict())      # type: ignore
target_qnet.to(DEVICE)

pnet = mccPNet(num_observations, num_actions)
if CONFIG['network']['p_load'] not in ["", "None", "NONE"]:
    pnet.load(CONFIG['network']['p_load'])
pnet.to(DEVICE)
pnet = pnet.float()

target_pnet = mccPNet(num_observations, num_actions)
target_pnet.load_state_dict(pnet.state_dict())      # type: ignore
target_pnet.to(DEVICE)

"""
Q_SAVE_LOCATION = CONFIG['network']['q_save']
assert os.path.exists(os.path.dirname(Q_SAVE_LOCATION)) and \
    (not os.path.exists(Q_SAVE_LOCATION) or not os.path.isdir(Q_SAVE_LOCATION)), \
        f"q_save {Q_SAVE_LOCATION} is not a valid path"
P_SAVE_LOCATION = CONFIG['network']['p_save']
assert os.path.exists(os.path.dirname(P_SAVE_LOCATION)) and \
    (not os.path.exists(P_SAVE_LOCATION) or not os.path.isdir(P_SAVE_LOCATION)), \
        f"p_save {P_SAVE_LOCATION} is not a valid path"
"""


# hyper parameters
HYPERP = CONFIG['hyperparameters']
EPOCHS = int(HYPERP['epochs'])
Q_LEARNING_RATE_INIT = float(HYPERP['q_lr_init'])
Q_LEARNING_RATE_FINAL = float(HYPERP['q_lr_final'])
P_LEARNING_RATE_INIT = float(HYPERP['p_lr_init'])
P_LEARNING_RATE_FINAL = float(HYPERP['p_lr_final'])
DISCOUNT = 0.99
NUM_P_STEPS_PER_EPOCH = 1   # number of times that the p network is optimized per epoch
TARGET_UPDATE = 1           # number of epochs before updating the target networks
TAU = float(HYPERP['tau'])
criterion = nn.SmoothL1Loss()
q_optim = optim.Adam(qnet.parameters(), lr=Q_LEARNING_RATE_INIT, weight_decay=0.01)
p_optim = optim.Adam(pnet.parameters(), lr=P_LEARNING_RATE_INIT)
q_scheduler = lr_scheduler.StepLR(q_optim, step_size=5, gamma=nthroot(Q_LEARNING_RATE_FINAL/Q_LEARNING_RATE_INIT, EPOCHS/5))
p_scheduler = lr_scheduler.StepLR(p_optim, step_size=5*NUM_P_STEPS_PER_EPOCH, gamma=nthroot(P_LEARNING_RATE_FINAL/P_LEARNING_RATE_INIT, EPOCHS/5))

def sigma_scheduler(epoch):
    M1 = int(CONFIG['hyperparameters']['M1'])
    M2 = int(CONFIG['hyperparameters']['M2'])
    END_VALUE = float(CONFIG['hyperparameters']['end_value'])
    if epoch < M1: return 1
    elif epoch < M2: return 1 - (epoch-M1) / (M2-M1) + END_VALUE
    else: return END_VALUE

# practical hyper parameters
BATCH_SIZE = 100
BATCHES_PER_ITER = 10
MEMORY = int(HYPERP['memory'])
MEMORY_RENEWAL = int(HYPERP['memory_renewal'])

memory = Memory(num_observations, num_actions, MEMORY+2000)


MONITOR_DATA : List = []            # item epoch: (epoch, num_episodes, avg_duration, avg_reward)
ROLLING_REWARD_THRESHOLD = -40      # every time this threshold is achieved, the networks are saved (and the threshold is increased by 10)


def run_and_save_episode(policy: mccPNet, epoch, mem: Memory):
    """run the environment with the current neural network and save the results to memory, return statistics"""
    total_reward = 0
    duration = 1
    policy.eval()

    observation = env.reset()
    done = False
    while not done:
        action = policy.select_action(torch.as_tensor(observation, device=DEVICE), sigma=sigma_scheduler(epoch)).cpu().detach()        # change sigma for epoch-dependent action selection
        star = torch.cat((torch.as_tensor(observation, device=DEVICE), torch.as_tensor(action, device=DEVICE)))
        observation, reward, done, _ = env.step(action)
        reward += -0.01+0.01*np.sin(3*observation[0])      # little hack: give an extra penalty for being low to the bottom
        star = torch.cat((star, torch.as_tensor(observation, device=DEVICE), torch.as_tensor(reward, device=DEVICE).view(1)))
        mem.add_star(star)
        
        duration += 1
        total_reward += reward

    # logger.debug((epoch, duration, float(total_reward)))

    return duration, float(total_reward)


def refill_memory(policy, epoch):
    """
    delete the oldest states of memory
    run enough episodes to refill the memory
    save statistics to MONITOR_DATA
    """

    durations = []
    total_rewards = []
    num_episodes = 0

    if len(memory) >= MEMORY:
        memory.remove_first_states(MEMORY_RENEWAL)

    while len(memory) < MEMORY:
        duration, total_reward = run_and_save_episode(policy, epoch, mem=memory)
        durations.append(duration)
        total_rewards.append(total_reward)
        num_episodes += 1

        if num_episodes%100 == 0:
            logger.info(f"episode nr {num_episodes}")
            logger.info(f"  memory length: {memory.length}")

    avg_duration = np.mean(durations) if num_episodes > 0 else MONITOR_DATA[epoch-1][2]
    avg_reward = np.mean(total_rewards) if num_episodes > 0 else MONITOR_DATA[epoch-1][3]
    logger.info('Number of episodes:'.center(23) + str(num_episodes))
    logger.info('Average duration:'.center(23) + str(avg_duration))
    logger.info('Average reward:'.center(23) + str(avg_reward))
    MONITOR_DATA.append((epoch, num_episodes, avg_duration, avg_reward))


def record_episode(policy: mccPNet, path, sigma=0):
    """run the environment (deterministically) with the current neural network and save the recording"""

    policy.eval()
    recorder = VideoRecorder(env=env, path=path, enabled=True)

    observation = env.reset()
    done = False
    while not done:
        recorder.capture_frame()
        action = policy.select_action(torch.as_tensor(observation, device=DEVICE), sigma=sigma).cpu().detach()        # change sigma for epoch-dependent action selection
        observation, _, done, _ = env.step(action)

    recorder.close()
    env.close()


def save_models_if_necessary(qnet, pnet):
    """
    calculate the rolling reward:
    if this exceeds a certain threshold, save the networks and update the threshold
    """

    global ROLLING_REWARD_THRESHOLD
    rolling_reward = sum(num_e*avg_rew for _,num_e,_,avg_rew in MONITOR_DATA[-10:]) / sum(num_e for _,num_e,_,_ in MONITOR_DATA[-10:])
    logger.info('Rolling reward:'.center(23) + str(rolling_reward))

    if rolling_reward >= ROLLING_REWARD_THRESHOLD:
        qnet.save(Q_BEST_LOCATION)
        pnet.save(P_BEST_LOCATION)
        ROLLING_REWARD_THRESHOLD += 10
        logger.info("Rolling reward exceeded threshold! Networks saved!")


def train_qmodel(q_func: mccQNet, q_target: mccQNet, p_target: mccPNet):
    # do one training cycle
    # return the total loss

    q_func.train()
    q_target.eval()
    p_target.eval()

    total_q = []
    fcl_grad_std = []    # save tuples (fcl1_grad_std, fcl2_grad_std, fcl3_grad_std)
    fcl3_abs_bias = []
    running_loss = 0.
    
    for transitions, next_states, rewards in memory.sample_tr_st_r(BATCH_SIZE, max_batches=BATCHES_PER_ITER):
        q_values = q_func(transitions)

        next_transitions = torch.hstack((next_states, p_target.select_action(next_states, sigma=0)))
        targets = rewards + DISCOUNT * q_target(next_transitions)           # TODO: do not add q_target(next_transitions) for terminal states

        loss = criterion(q_func.normalize(q_values), q_func.normalize(targets))

        q_optim.zero_grad()
        loss.backward()
        q_optim.step()

        running_loss += loss.item()
        total_q.append(torch.mean(q_values).item())
        fcl_grad_std.append((
            torch.std(q_func.fcl1.weight.grad).item(), 
            torch.std(q_func.fcl2.weight.grad).item(),
            torch.std(q_func.fcl3.weight.grad).item(),
        ))
        fcl3_abs_bias.append(np.abs(qnet.fcl3.bias.grad.item()))

    q_scheduler.step()

    logger.info("average Q: %s", np.mean(total_q))
    logger.debug("qnet grad std by layer: %s", np.mean(fcl_grad_std, axis=0))
    logger.debug("average fcl3 bias grad: %s", np.mean(fcl3_abs_bias))

    logger.debug(f"qloss: {running_loss}")


def train_pmodel(policy: mccPNet, q_target: mccQNet):
    # do one training cycle
    # return the total loss

    policy.train()
    q_target.eval()

    fcl_grad_std = []    # save tuples (fcl1_grad_std, fcl2_grad_std, fcl3_grad_std)
    running_loss = 0.

    for states in memory.sample_states(BATCH_SIZE, max_batches=BATCHES_PER_ITER):
        actions = policy(states)
        transitions = torch.hstack((states, actions))
        q_vector = q_target(transitions)
        loss = -torch.mean(q_target.normalize(q_vector), dim=0)     # let the cost be the opposite of the q values of the generated transitions, as our goal is to maximize this q-value
        running_loss += loss.item()

        p_optim.zero_grad()
        loss.backward()
        p_optim.step()

        fcl_grad_std.append((
            torch.std(policy.fcl1.weight.grad).item(), 
            torch.std(policy.fcl2.weight.grad).item(),
            torch.std(policy.fcl3.weight.grad).item(),
        ))

    p_scheduler.step()

    logger.debug(f"ploss: {running_loss}")
    logger.debug("pnet grad std by layer: %s", np.mean(fcl_grad_std, axis=0))



def main():
    try:
        for epoch in range(EPOCHS):
            logger.info(f"\nstarting epoch {epoch+1}/{EPOCHS}")

            if epoch>0 and epoch%100 == 0:
                record_episode(policy=pnet, path=os.path.join(VIDEO_BASE_PATH, f"epoch {epoch} det.mp4"), sigma=0)
                record_episode(policy=pnet, path=os.path.join(VIDEO_BASE_PATH, f"epoch {epoch} stoch.mp4"), sigma=sigma_scheduler(epoch))

            refill_memory(pnet, epoch)
            save_models_if_necessary(qnet=qnet, pnet=pnet)

            logger.info('P_lr: %s', p_scheduler.get_last_lr())
            logger.info('Q_lr: %s', q_scheduler.get_last_lr())

            # train models
            for i in range(NUM_P_STEPS_PER_EPOCH):
                train_pmodel(pnet, target_qnet)
            train_qmodel(qnet, target_qnet, target_pnet)

            if epoch % TARGET_UPDATE == 0:
                soft_update(target_qnet, qnet, TAU)
                soft_update(target_pnet, pnet, TAU)
    except KeyboardInterrupt:
        pass
    
    qnet.save(Q_SAVE_LOCATION)
    pnet.save(P_SAVE_LOCATION)
    logger.info(f"Qnet saved to {Q_SAVE_LOCATION}")
    logger.info(f"Pnet saved to {P_SAVE_LOCATION}")
    with open(os.path.join(TASK_DIR, 'task_data'), 'w') as f:
        f.write(str(MONITOR_DATA))

    plot_rewards(MONITOR_DATA)
    

if __name__ == "__main__":
    main()
