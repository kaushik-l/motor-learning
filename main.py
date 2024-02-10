from model import Network, Task, Algorithm
from train import learn, follow
import numpy.random as npr

# net, task, algo, learning = train('learner', N=128, S=0, R=1, g=2,
#                                   task='motor', duration=20, cycles=1,
#                                   algo='reinforce', Nepochs=300000, lr=1, noise=0.005, clip=1e-4)

net, task, algo, stats = learn('learner', N=128, S=0, R=1, g=2,
                               task='reach_1d', duration=20, Nepochs=100000,
                               algo='reinforce', lr=1, noise=0.04, clip=1e-4)

# net, task, algo, stats = follow(N=128, S=0, R=1, g=2,
#                                 task='reach_1d', duration=20, Nepochs=10000,
#                                 algos=('reinforce', 'rflo'),
#                                 lr=(1, 1e-4), noise=(0.01, 0), clip=1e-4)
