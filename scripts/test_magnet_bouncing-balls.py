import time
import os
import argparse
import numpy as np
import pylab as plt
import torch
import torch.nn as nn

import sys
sys.path.append('./')

from models.MagNet2 import MagNet
from dataGen.bouncing_balls import bouncing_balls

parser = argparse.ArgumentParser(description='Paths and switches')
parser.add_argument('--seed', default=2025, help='Seed for random numbers')
parser.add_argument('--model_path', default=None, help='Path to any pretrained model')
parser.add_argument('--wrapper_path', default=None, help='Path to any pretrained wrapper')
parser.add_argument('--savepath', default=None, help='Path to save results')
parser.add_argument('--musigma', default=None, help='Path to trainset musigma If noise is True')
args = parser.parse_args()

torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Time step
dt = 0.01

# Get means and stds of training set
n_objects = 5
seq_len = 16

dataset = bouncing_balls(n_objects=n_objects)

if args.musigma is None:
    n_samples = 500
    n_frames = 500
    states, _, _ = dataset.generate_data(n_samples=n_samples,
                                         n_frames=n_frames,
                                         seed=0,
                                         dt=dt)

    means = np.mean(states, axis=(0,1,2))
    stds = np.std(states, axis=(0,1,2))
    means = means[np.newaxis,:]
    stds = stds[np.newaxis,:]
else:
    musigma = np.load(args.musigma)
    means = musigma[:-1]
    stds = musigma[1:]


# Load model
magnet = MagNet(n_objects=n_objects)
magnet = magnet.cuda()
checkpoint = torch.load(args.model_path)
magnet.load_state_dict(checkpoint)
checkpoint = torch.load(args.wrapper_path)
magnet.I3 = checkpoint['I3']
magnet.S2W = checkpoint['S2W']
magnet.S2b = checkpoint['S2b']
magnet = magnet.eval()

# Test on a video
states, _, _ = dataset.generate_data(n_samples=1,
                                     n_frames=500,
                                     seed=int(args.seed),
                                     dt=dt)

states = states.reshape((500, n_objects, 4))
ground_truth = states.copy()

means = torch.from_numpy(means).float().cuda()
stds = torch.from_numpy(stds).float().cuda()

t = seq_len//2
GTs = []
Prds = []
with torch.no_grad():
    while t < seq_len//2 + 200:
        if t == seq_len//2:
            X = states[t:t+1]
            X = torch.from_numpy(X).float().cuda()
            X = (X - means) / stds
        else:
            X = y_pred
        y = ground_truth[t+1:t+2]

        y_pred = torch.zeros((1, n_objects, 4))
        dX = magnet(X)
        y_pred[:,:,2:] = X[:,:,2:] + dX*dt
        y_pred[:,:,:2] = X[:,:,:2] + ((stds[:,2:]*y_pred[:,:,2:] + means[:,2:])/stds[:,:2])*dt

        truth = y[0,:,:2]
        pred = y_pred[0,:,:2]*stds[:,:2] + means[:,:2]
        pred = pred.cpu().numpy()

        if args.savepath is not None:
            dataset.visualize(truth, t, savepath=os.path.join(args.savepath, 'true/'))
            dataset.visualize(pred, t, savepath=os.path.join(args.savepath, 'predicted/'))

        GTs.append(truth)
        Prds.append(pred)

        t += 1

GTs = np.array(GTs)
Prds = np.array(Prds)
if args.savepath is not None:
    dataset.plot_trajectory(GTs, savepath=args.savepath + 'gt.png')
    dataset.plot_trajectory(Prds, savepath=args.savepath + 'pred.png')
