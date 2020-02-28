import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append('./')

from models.MagNet import MagNet
from dataGen.swarm import swarm
from utils.utils import *

parser = argparse.ArgumentParser(description='Paths and switches')
parser.add_argument('--model_path', default=None, help='Path to any pretrained model')
parser.add_argument('--wrapper_path', default=None, help='Path to any pretrained wrapper')
parser.add_argument('--noise', default=False, help='Add noise?')
args = parser.parse_args()

torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Time step
dt = 0.01

# Load train data
n_predator=1 
n_prey=20
n_objects = n_predator + n_prey
seq_len = 16

dataset = swarm(n_predator=1, n_prey=20)

n_samples = 100
n_frames = 300
print("Generating dataset...")
states, _ = dataset.generate_data(n_samples=n_samples,
                                  n_frames=n_frames,
                                  seed=0,
                                  dt=dt)

if args.noise:
    np.random.seed()
    states = add_noise(states, idxs=[0,1], diff=True, dt=dt)

sequences, shifted_sequences = generate_sequences(states, seq_len)

means = np.mean(sequences, axis=(0,1,2))
stds = np.std(sequences, axis=(0,1,2))
means = means[np.newaxis,:]
stds = stds[np.newaxis,:]

np.save('./saved_models/swarm/trainset_musigma.npy', np.concatenate([means, stds], axis=0))

train_count = len(sequences)
order = np.random.choice(train_count, train_count, replace=False)
train_data = {}
train_data['X'] = sequences[order]
train_data['y'] = shifted_sequences[order]

# Standardize training data
train_data['X'] = (train_data['X'] - means)/stds
train_data['y'] = (train_data['y'] - means)/stds

# Load validation data
n_samples = 10
n_frames = 500
states, _ = dataset.generate_data(n_samples=n_samples,
                                  n_frames=n_frames,
                                  seed=200,
                                  dt=dt)

if args.noise:
    np.random.seed()
    states = add_noise(states, idxs=[0,1], diff=True, dt=dt)

sequences, shifted_sequences = generate_sequences(states, seq_len)
val_count = len(sequences)
val_data = {}
val_data['X'] = sequences
val_data['y'] = shifted_sequences

# Standardize validation data
val_data['X'] = (val_data['X'] - means)/stds
val_data['y'] = (val_data['y'] - means)/stds

# Define model
magnet = MagNet(n_objects=n_objects)
magnet = magnet.cuda()
if args.model_path is not None:
    checkpoint = torch.load(args.model_path)
    model_dict = magnet.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    magnet.load_state_dict(model_dict)
if args.wrapper_path is not None:
    checkpoint = torch.load(args.wrapper_path)
    magnet.I3 = checkpoint['I3']
    magnet.S2W = checkpoint['S2W']
    magnet.S2b = checkpoint['S2b']

# Define optimizer and loss
optimizer = optim.Adam([
                       {'params': magnet.parameters()},
                       {'params': [magnet.I3, magnet.S2W, magnet.S2b], 'lr': 1e-3}
                       ], lr=1e-3)               

criterion = nn.SmoothL1Loss()

means = torch.from_numpy(means).float().cuda()
stds = torch.from_numpy(stds).float().cuda()

# Train
print("Starting Training\n")
num_epoches = 100
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

        train_error += train_loss.item()

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
    # Exponential decay learning rate        
    for g in optimizer.param_groups:
        if g['lr'] > 0.0001:
            g['lr'] *= 0.95
        print("LR: {}".format(g['lr']))
    # Save models    
    torch.save(magnet.state_dict(), 'saved_models/swarm/model-{}.ckpt'.format(epoch))
    dict = {}
    dict['I3'] = magnet.I3
    dict['S2W'] = magnet.S2W
    dict['S2b'] = magnet.S2b
    torch.save(dict, 'saved_models/swarm/wrapper-{}.ckpt'.format(epoch))



