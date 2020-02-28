import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append('./')

from models.MagNet import MagNet
from dataGen.point_masses import point_masses
from utils.utils import *

parser = argparse.ArgumentParser(description='Paths and switches')
parser.add_argument('--model_path', default=None, help='Path to any pretrained model')
parser.add_argument('--wrapper_path', default=None, help='Path to any pretrained wrapper')
parser.add_argument('--pretrained_agents', default=None, help='Number of agents in pretrained wrapper')
parser.add_argument('--musigma', default=None, help='Path to trainset mean and std.dev')

args = parser.parse_args()

torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Time step
dt = 0.01

# Load data
n_objects = 8
seq_len = 16

dataset = point_masses(n_objects=n_objects)

n_frames = 11000
states, _ = dataset.generate_data(n_samples=1,
                                  n_frames=n_frames,
                                  seed=int(args.seed),
                                  dt=dt)

musigma = np.load(args.musigma)
means = musigma[:-1]
stds = musigma[1:]
means = torch.from_numpy(means).float().cuda()
stds = torch.from_numpy(stds).float().cuda()

# Load model
magnet = MagNet(n_objects=n_objects)
magnet = magnet.cuda()
checkpoint = torch.load(args.model_path)
model_dict = magnet.state_dict()
pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
model_dict.update(pretrained_dict)
magnet.load_state_dict(model_dict)
for name, param in magnet.named_parameters():
    if name in pretrained_dict:
        param.requires_grad = False
checkpoint = torch.load(args.wrapper_path)
if int(args.pretrained_agents) == n_objects:
    magnet.I3 = checkpoint['I3']
    magnet.S2W = checkpoint['S2W']
    magnet.S2b = checkpoint['S2b']
else:
    magnet.I3[:].data = torch.mean(checkpoint['I3'], dim=0)
    magnet.S2W[:].data = torch.mean(checkpoint['S2W'], dim=0)
    magnet.S2b[:].data = torch.mean(checkpoint['S2b'], dim=0)

# Predict and re-tune
t = 0
while t < (n_frames-1):
    X = states[0,t:t+1]
    X = torch.from_numpy(X).float().cuda()
    X = (X - means) / stds
    error = 0.
    while error < 20:
        with torch.no_grad():
            y = states[0,t+1:t+2]

            y_pred = torch.zeros((1, n_objects, 4))
            dX = magnet(X)
            y_pred[:,:,2:] = X[:,:,2:] + dX*dt
            y_pred[:,:,:2] = X[:,:,:2] + ((stds[:,2:]*y_pred[:,:,2:] + means[:,2:])/stds[:,:2])*dt

            truth = y[0,:,:2]
            pred = y_pred[0,:,:2]*stds[:,:2] + means[:,:2]
            pred = pred.cpu().numpy()

            error = np.mean(np.square(truth - pred))
            print('Frame: {}, Error: {}'.format(t, error))

            t += 1
            X = y_pred

    print("Re-tuning required\n")

    sequences, shifted_sequences = generate_sequences(states[:,t:t+10000], seq_len)
    means = np.mean(sequences, axis=(0, 1, 2))
    stds = np.std(sequences, axis=(0, 1, 2))
    means = means[np.newaxis, :]
    stds = stds[np.newaxis, :]

    train_count = len(sequences)
    order = np.random.choice(train_count, train_count, replace=False)
    train_data = {}
    train_data['X'] = sequences[order]
    train_data['y'] = shifted_sequences[order]

    # Standardize training data
    train_data['X'] = (train_data['X'] - means) / stds
    train_data['y'] = (train_data['y'] - means) / stds

    sequences, shifted_sequences = generate_sequences(states[:,t+10000:t+10100], seq_len)
    val_count = len(sequences)
    val_data = {}
    val_data['X'] = sequences
    val_data['y'] = shifted_sequences

    # Standardize validation data
    val_data['X'] = (val_data['X'] - means)/stds
    val_data['y'] = (val_data['y'] - means)/stds

    optimizer = optim.Adam([magnet.I3, magnet.S2W, magnet.S2b], lr=5e-4)

    criterion = nn.L1Loss(size_average=True)

    means = torch.from_numpy(means).float().cuda()
    stds = torch.from_numpy(stds).float().cuda()

    # Train
    print("Starting re-tuning\n")
    num_epoches = 30
    train_steps = train_count
    for epoch in range(num_epoches):
        train_error = 0.
        for step in range(train_steps):
            X = train_data['X'][step]
            y = train_data['y'][step]
            X = torch.from_numpy(X).float().cuda()
            y = torch.from_numpy(y).float().cuda()
            dX = magnet(X)
            vel_pred = X[:,:,2:] + dX*dt
            pos_pred = X[:,:,:2] + ((stds[:,2:]*vel_pred + means[:,2:])/stds[:,:2])*dt
            y_pred = torch.cat((pos_pred, vel_pred), dim=-1)
            train_loss = criterion(y_pred, y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_error += train_loss.clone().detach()

            if (step+1)%5000 == 0:
                with torch.no_grad():
                    val_error = 0.
                    for count in range(val_count):
                        val_X = val_data['X'][count]
                        val_y = val_data['y'][count]
                        val_X = torch.from_numpy(val_X).float().cuda()
                        val_y = torch.from_numpy(val_y).float().cuda()
                        val_y_pred = torch.zeros((seq_len, n_objects, 4))
                        val_dX = magnet(val_X)
                        val_y_pred[:,:,2:] = val_X[:,:,2:] + val_dX*dt
                        val_y_pred[:,:,:2] = val_X[:,:,:2] + ((stds[:,2:]*val_y_pred[:,:,2:] + means[:,2:])/stds[:,:2])*dt
                        val_loss = criterion(val_y_pred, val_y)
                        val_error += val_loss

                print("Step: {}, Training Error: {}, Validation Error: {}".format(epoch*train_steps+step+1,
                                                                                  train_error/5000,
                                                                                  val_error/val_count))
                train_error = 0.
        for g in optimizer.param_groups:
            if g['lr'] > 0.0001:
                g['lr'] *= 0.95
            print("LR: {}".format(g['lr']))

    t += 10100


