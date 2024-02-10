import os
import pickle

import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt


def movmedian(x, winsize, adapt=True):
    if adapt:
        x_median = [np.median(x[idx - np.min((idx, winsize)):idx]) for idx in np.arange(len(x))]
    else:
        x_median = [np.median(x[idx-winsize:idx]) for idx in np.arange(winsize, len(x))]
    return x_median


# flags
motor__reinforce_rflo = False
motor_reinforce_vary__noise = True
motor_reinforce_vary__noise_clip = False

if motor__reinforce_rflo:
    # initialize mses
    learner_mses, follower_mses = [], []
    learner_xa, follower_xa = [], []
    fnames = [f for f in os.listdir('Data//' + 'motor_reinforce_rflo' + '') if not f.startswith('.')]
    count = 1
    for f in fnames:
        file = open('.//Data//motor_reinforce_rflo//' + f, 'rb')
        data = pickle.load(file)
        file.close()
        datacopy = data
        net, task, algo, stats = datacopy['net'], datacopy['task'], datacopy['algo'], datacopy['stats']
        learner_mses.append(np.array(stats['learner']['mses_test']))
        learner_xa.append(np.array(stats['learner']['xa']))
        follower_mses.append(np.array(stats['follower']['mses_test']))
        follower_xa.append(np.array(stats['follower']['xa']))
        print('\r file ' + str(count) + '/' + str(len(fnames)), end='')
        count += 1
    # learning curve
    plt.errorbar(np.arange(30000), np.median(np.array(learner_mses), axis=0),
                 np.std(np.array(learner_mses), axis=0) / np.sqrt(1000), elinewidth=.01, label='REINFORCE')
    plt.errorbar(np.arange(30000), np.median(np.array(follower_mses), axis=0),
                 np.std(np.array(follower_mses), axis=0) / np.sqrt(1000), elinewidth=.01, label='RFLO')
    plt.yscale('log'), plt.legend()
    plt.xlabel('Trial', fontsize=18), plt.ylabel('MSE', fontsize=18)
    plt.show()
    # trajectories
    plt.subplot(2, 1, 1)
    plt.plot(np.array(learner_xa)[::50, :, 0].T, 'b'), plt.ylim([0, 40])
    plt.xlabel('Time', fontsize=18), plt.ylabel('Position', fontsize=18)
    plt.subplot(2, 1, 2)
    plt.plot(np.array(follower_xa)[npr.random_integers(1000), :, 0].T, 'r'), plt.ylim([0, 40])
    plt.xlabel('Time', fontsize=18), plt.ylabel('Position', fontsize=18)
    plt.show()


if motor_reinforce_vary__noise:
    # initialize mses
    noise, mses = [], []
    fnames = [f for f in os.listdir('Data//motor_reinforce//vary__noise') if not f.startswith('.')]
    count = 1
    for f in fnames:
        file = open('.//Data//motor_reinforce//vary__noise//' + f, 'rb')
        data = pickle.load(file)
        file.close()
        net, task, algo, stats = data['net'], data['task'], data['algo'], data['stats']
        noise.append(algo.noise)
        mses.append(np.array(stats['mses_test']))
        print('\r file ' + str(count) + '/' + str(len(fnames)), end='')
        count += 1
    noise_list = np.unique(noise)
    mses_mean = []
    for n in noise_list:
        mses_mean.append(np.mean(np.array(mses)[noise == n], axis=0))
    mse1 = np.array(mses_mean)[:4, :].mean(axis=0)
    mse2 = np.array(mses_mean)[4:8, :].mean(axis=0)
    mse3 = np.array(mses_mean)[8:12, :].mean(axis=0)
    mse4 = np.array(mses_mean)[12:16, :].mean(axis=0)
    plt.plot(movmedian(mse1, 100))
    plt.plot(movmedian(mse2, 100))
    plt.plot(movmedian(mse3, 100))
    plt.plot(movmedian(mse4, 100))
    plt.yscale('log')
    plt.show()
