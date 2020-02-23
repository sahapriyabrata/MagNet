import numpy as np

from utils.tvregdiff import TVRegDiff

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

def add_noise(states, idxs, diff=True, dt=0.01):
    n_samples, n_frames, n_objects = states.shape[:3]
    stds = np.std(states, axis=(0, 1, 2))
    stds = stds[np.newaxis, :]
    states[:,:,:,idxs] += 0.005*stds[:,idxs]*np.random.randn(n_samples, n_frames, n_objects, len(idxs))
    if diff:
        for n in range(n_samples):
            for k in range(n_objects):
                for d in range(len(idxs)):
                    states[n,:,k,len(idxs)+d] = TVRegDiff(states[n,:,k,d], 100, 1e-1, dx=dt,
                                                          ep=1e-2, scale='large', plotflag=0)
    return states


