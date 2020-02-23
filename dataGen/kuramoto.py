import numpy as np
import matplotlib as mpl
import pylab as plt

class kuramoto:
    def __init__(self, n_objects=5):
        self.n_objects = n_objects
    
    def generate_data(self, n_samples=1, n_frames=1000, dt=0.01, seed=0):
        states = np.zeros((n_samples, n_frames, self.n_objects, 1), dtype=np.float)
        shifted_states = np.zeros((n_samples, n_frames, self.n_objects, 1), dtype=np.float)
        
        omega = [w for w in range(self.n_objects)]
        omega= 0.5*np.array(omega)
        for n in range(n_samples):
            np.random.seed(seed+n)
            theta = 2*np.pi*np.random.rand(self.n_objects,1)
            states[n,0] = theta
            for t in range(n_frames):
                temp = theta
                for i in range(self.n_objects):
                    dtheta = 0.
                    for j in range(self.n_objects):
                        K = 0.05*(i+j)
                        dtheta += K*np.sin(temp[j] - temp[i])
                    dtheta += omega[i]
                    theta[i] += dtheta*dt
                shifted_states[n,t] = theta
            states[n,1:,] = shifted_states[n,:-1,]
        return states, shifted_states
                    
    def plot_trajectory(self, phase, savepath='results_imgs/'):
        timesteps = np.arange(200)
        for i in range(phase.shape[1]):
            plt.plot(timesteps, np.remainder(phase[:200,i], 2*np.pi))
        plt.ylabel('Phase')
        plt.xlabel('Timesteps')
        plt.savefig(savepath)
        plt.close()
        
    def visualize(self, states, i, savepath='results_imgs/'):
        circle = plt.Circle((0,0), 64, edgecolor='r', facecolor='w')
        ax = plt.gca()
        ax.axis('equal')
        plt.plot(states[:,0], states[:,1], 'co', ms = 15)
        ax.add_patch(circle)
        plt.axis([-128, 128, -128, 128])
        plt.axis('off')
        plt.savefig(savepath+'%i_animate.png' % (i + 1))
        plt.close()    



