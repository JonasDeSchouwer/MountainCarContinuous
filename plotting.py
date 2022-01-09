"""
We will try to plot data in the form of a list of tuples (epoch, duration, reward)
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_durations(data):
    """
    :param data: list of tuples [(epoch, duration)]
    """

    epochs, durations, _ = zip(*data)
    EPOCHS = max(epochs)
    epochs = np.array(epochs)
    durations = np.array(durations)

    plt.scatter(epochs, durations, c='r')

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


def plot_rewards(data):
    """
    :param data: list of tuples [(epoch, duration)]
    """

    epochs, _, rewards = zip(*data)
    EPOCHS = max(epochs)
    epochs = np.array(epochs)
    rewards = np.array(rewards)

    plt.scatter(epochs, rewards, c='r')

    avg_reward = np.zeros(EPOCHS)       # avg_reward[i] = gemiddelde reward in epoch i
    for epoch in range(EPOCHS):
        elements = rewards[epochs==epoch]
        if len(elements) == 0:
            avg_reward[epoch] = avg_reward[epoch-1]
        else:
            average_dur = np.mean(elements)     #get those columns for which the first element is epoch
            avg_reward[epoch] = average_dur

    # take a rolling mean of avg_reward
    D = 10
    smooth_reward = np.zeros(EPOCHS)
    for i in range(EPOCHS):
        start = max(i-D, 0)
        end = min(i+D,EPOCHS)
        smooth_reward[i] = np.mean(avg_reward[start:end])

    plt.plot(smooth_reward, 'b')

    plt.show()