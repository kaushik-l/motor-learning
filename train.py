import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt
from model import Network, Task, Algorithm
import scipy.linalg as linalg
from sklearn.decomposition import PCA
from scipy.stats.stats import pearsonr


def movmean(x, winsize, adapt=True):
    if adapt:
        x_mean = [x[idx - np.min((idx, winsize)):idx].mean() for idx in np.arange(len(x))]
    else:
        x_mean = [x[idx-winsize:idx].mean() for idx in np.arange(winsize, len(x))]
    return x_mean


def movmedian(x, winsize, adapt=True):
    if adapt:
        x_median = [np.median(x[idx - np.min((idx, winsize)):idx]) for idx in np.arange(len(x))]
    else:
        x_median = [np.median(x[idx-winsize:idx]) for idx in np.arange(winsize, len(x))]
    return x_median


def learn(name='rnn', N=256, S=0, R=1, g=1.5, task='motor', duration=20, cycles=2,
          algo='reinforce', Nepochs=10000, lr=1e-1, noise=0.1, clip=1e-4, seed=1):

    # instantiate model
    npr.seed(seed)
    net = Network(name, N, S, R, g=g, seed=seed)
    task = Task(task, duration, cycles)
    algo = Algorithm(algo, Nepochs, lr, noise, clip)

    # frequently used vars
    dt, NT, N, S, R = net.dt, task.NT, net.N, net.S, net.R

    # track variables during learning
    stats = {'lr': [], 'mses': [], 'mses_test': [], 'reward': [], 'ua': []}

    # random initialization of hidden state
    z0 = 0.1 * npr.randn(N, 1)    # hidden state (potential)
    net.z0 = z0  # save

    # initialize expected error
    mse_mean = 1

    for ei in range(algo.Nepochs):

        # initialize activity
        z0 = net.z0  # hidden state (potential)
        h0 = net.f(z0)  # hidden state (rate)
        z, h = z0, h0

        # initialize output
        v, x = np.zeros((1, 1)), np.zeros((1, 1))

        # save tensors for plotting
        ha = np.zeros((NT, N))  # save the hidden states for each time bin for plotting
        ua = np.zeros((NT, R))  # save acc
        va = np.zeros((NT, R))  # save vel
        xa = np.zeros((NT, R))  # save pos

        # errors
        err = np.zeros((NT, R))     # errors

        # eligibility trace
        e = np.zeros((N, N))

        for ti in range(NT):

            # network update
            z = np.matmul(net.J, h)     # potential

            # update activity
            h = (1 - dt) * h + dt * (net.f(z))  # activity

            # add perturbative noise
            delta = algo.noise * (2 * npr.random((N, 1)) - 1)
            h += delta

            # update eligibility trace
            e += np.matmul(delta, z.T)

            # generate output
            u = np.matmul(net.wr, h)  # output
            v += u * dt
            x += v * dt

            # save values for plotting
            ha[ti], ua[ti], va[ti], xa[ti] = h.T, u.T, v.T, x.T

            # error
            err[ti] = task.ustar[ti] - u

        # print loss
        mse = task.loss(err)
        # mse = np.sqrt(((task.xstar[-50:] - xa[-50:].squeeze()) ** 2).mean() / 2)
        if (ei+1) % 100 == 0:
            print('\r' + str(ei + 1) + '/' + str(algo.Nepochs) + '\t Err:' + str(mse) + '\t Reward:' + str(mse_mean - mse), end='')

        # test loss
        mse_test = np.mean((task.xstar[-25:] - xa[-25:]) ** 2)

        # update weights
        reward = mse_mean - mse
        net.J += np.clip(algo.lr * e * reward, -algo.clip, algo.clip)

        # update mean error
        mse_mean = 0.1*mse_mean + 0.9*mse

        # save mse list for each epoch
        stats['mses'].append(mse)
        stats['mses_test'].append(mse_test)
        stats['reward'].append(reward)

    # save other variables
    stats['lr'] = lr
    stats['ua'] = ua
    stats['xa'] = xa

    return net, task, algo, stats


def follow(N=256, S=0, R=1, g=1.5, task='motor', duration=20, cycles=2,
           algos=('reinforce', 'rflo'), Nepochs=10000, lr=(1e-1, 1e-1), noise=(0.1, 0), clip=1e-4, seed=1):

    # instantiate model
    npr.seed(seed)
    net, algo, stats = {}, {}, {}
    net['learner'] = Network('learner', N, S, R, g=g, seed=seed)
    net['follower'] = Network('follower', N, S, R, g=g, seed=seed+1)
    task = Task(task, duration, cycles)
    algo['learner'] = Algorithm(algos[0], Nepochs, lr[0], noise[0], clip)
    algo['follower'] = Algorithm(algos[1], Nepochs, lr[1], noise[1], clip)

    # frequently used vars
    dt, NT, N, S, R = task.dt, task.NT, net['learner'].N, net['learner'].S, net['learner'].R

    # track variables during learning
    stats['learner'] = {'mses': [], 'reward': [], 'ua': [], 'xa': [], 'mses_test': []}
    stats['follower'] = {'mses': [], 'ua': [], 'xa': [], 'mses_test': []}

    # random initialization of hidden state
    z0 = 0.1 * npr.randn(N, 1)    # hidden state (potential)
    net['learner'].z0 = z0  # save
    net['follower'].z0 = z0  # save

    # initialize expected error
    mse_mean = 1

    for ei in range(Nepochs):

        # initialize activity
        z0 = net['learner'].z0  # hidden state (potential)
        h0 = net['learner'].f(z0)  # hidden state (rate)
        zl, hl = z0, h0   # learner
        zf, hf = z0, h0   # follower

        # initialize output
        vl, xl, vf, xf = np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))

        # save tensors for plotting
        hal, haf = np.zeros((NT, N)), np.zeros((NT, N))  # save the hidden states for each time bin for plotting
        ual, uaf = np.zeros((NT, R)), np.zeros((NT, R))  # save acc
        val, vaf = np.zeros((NT, R)), np.zeros((NT, R))  # save vel
        xal, xaf = np.zeros((NT, R)), np.zeros((NT, R))  # save pos

        # errors
        errl = np.zeros((NT, R))        # learner error
        errf = np.zeros((NT, R))        # follower error (training)
        errf2 = np.zeros((NT, R))       # follower error (testing)

        # eligibility trace
        el = np.zeros((N, N))           # for learner
        ef = np.zeros((N, N))           # for follower

        for ti in range(NT):

            # network update
            zl = np.matmul(net['learner'].J, hl)
            zf = np.matmul(net['follower'].J, hf)

            # update activity
            hl = (1 - dt) * hl + dt * (net['learner'].f(zl))  # learner
            hf = (1 - dt) * hf + dt * (net['follower'].f(zf))  # follower

            # add perturbative noise
            deltal = algo['learner'].noise * (2 * npr.random((N, 1)) - 1)
            hl += deltal
            deltaf = algo['follower'].noise * (2 * npr.random((N, 1)) - 1)
            hf += deltaf

            # update eligibility trace
            el += np.matmul(deltal, zl.T)
            ef = dt * net['follower'].df(zf) * hf.T + (1 - dt) * ef

            # generate output (learner)
            ul = np.matmul(net['learner'].wr, hl)  # output
            vl += ul * dt
            xl += vl * dt
            # generate output (follower)
            uf = np.matmul(net['follower'].wr, hf)  # output
            vf += uf * dt
            xf += vf * dt

            # save values for plotting
            hal[ti], ual[ti], val[ti], xal[ti] = hl.T, ul.T, vl.T, xl.T
            haf[ti], uaf[ti], vaf[ti], xaf[ti] = hf.T, uf.T, vf.T, xf.T

            # training error
            errl[ti] = task.ustar[ti] - ul      # learner error
            errf[ti] = ul - uf                  # follower error (training)
            errf2[ti] = task.ustar[ti] - uf     # follower error (testing)

            # update weights (follower)
            net['follower'].J += ((algo['follower'].lr / NT) * np.matmul(net['follower'].B, errf[ti]).reshape(N, 1) * ef)
            net['follower'].wr += (((algo['follower'].lr / NT) * hf) * errf[ti]).T
            net['follower'].B = net['follower'].wr.T * np.sqrt(N / R)

        # training loss
        msel = task.loss(errl)
        msef = task.loss(errf2)
        if (ei+1) % 100 == 0:
            print('\r' + str(ei + 1) + '/' + str(Nepochs) + '\t Err learner:' + str(msel) + '\t Err follower:' + str(msef), end='')

        # test loss
        msel_test = np.mean((task.xstar[-25:] - xal[-25:]) ** 2)
        msef_test = np.mean((task.xstar[-25:] - xaf[-25:]) ** 2)

        # update weights (learner)
        reward = mse_mean - msel
        net['learner'].J += np.clip(algo['learner'].lr * el * reward, -algo['learner'].clip, algo['learner'].clip)

        # update mean learner error
        mse_mean = 0.1*mse_mean + 0.9*msel

        # save mse list for each epoch
        stats['learner']['mses'].append(msel)
        stats['learner']['reward'].append(reward)
        stats['follower']['mses'].append(msef)
        stats['learner']['mses_test'].append(msel_test)
        stats['follower']['mses_test'].append(msef_test)

    # save other variables
    stats['learner']['ua'] = ual
    stats['learner']['xa'] = xal
    stats['follower']['ua'] = uaf
    stats['follower']['xa'] = xaf

    return net, task, algo, stats
