from train import learn, follow
import sys
import numpy.random as npr
import pickle

modelname = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1

if modelname == 'motor_reinforce':
    # train motor task using reinforce algorithm
    net, task, algo, stats = learn('learner', N=128, S=0, R=1, g=2,
                                   task='reach_1d', duration=20, Nepochs=200000,
                                   algo='reinforce', lr=1, noise=0.0025*npr.randint(1, 17),
                                   clip=1e-4, seed=seed)

if modelname == 'motor_reinforce_rflo':
    # train motor task using reinforce and rflo in parallel systems
    net, task, algo, stats = follow(N=128, S=0, R=1, g=2,
                                    task='reach_1d', duration=20, Nepochs=30000,
                                    algos=('reinforce', 'rflo'),
                                    lr=(1, 1e-4), noise=(0.01, 0), clip=1e-4, seed=seed)


# save
with open('//burg//theory//users//jl5649//motor-learning//' + modelname + '//' + str(seed) + '.pkl', 'wb') as f:
    pickle.dump({'net': net, 'task': task, 'algo': algo, 'stats': stats}, f)
