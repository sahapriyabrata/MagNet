import numpy as np
import cv2

from tvregdiff import TVRegDiff

def find_source_centers(power):
    thresh = np.uint8(power * 255)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    states = []
    for cnt in contours:
        (ypos, xpos), radius = cv2.minEnclosingCircle(cnt)
        states.append([xpos, ypos, radius])
    states = np.array(states)
    return states

def generate_sequences(states, seq_len):
    n_samples, n_frames, n_objects = states.shape[:3]
    sequences = []
    shifted_sequences = []
    for i in range(n_samples):
        for j in range(n_frames - seq_len - 1):
            sequences.append(states[i, j:j+seq_len])
            shifted_sequences.append(states[i, j+1:j+1+seq_len])

    sequences = np.array(sequences)
    shifted_sequences = np.array(shifted_sequences)
    return sequences, shifted_sequences

def add_noise(states, dt=0.01):
    n_samples, n_frames, n_objects = states.shape[:3]
    stds = np.std(states, axis=(0, 1, 2))
    stds = stds[np.newaxis, :]
    states[:,:,:,:2] += 0.005*stds[:,:2]*np.random.randn(n_samples, n_frames, n_objects, 2)
    for n in range(n_samples):
        for k in range(n_objects):
            for d in range(2):
                states[n,:,k,2+d] = TVRegDiff(states[n,:,k,d], 100, 1e-1, dx=dt,
                                              ep=1e-2, scale='large', plotflag=0)
    return states


