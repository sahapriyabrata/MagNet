import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class point_masses:
    def __init__(self, n_objects=4):
        self.n_objects = n_objects
        
    def generate_data(self, n_samples=1, n_frames=1000, dt=0.01, seed=0):
        states = np.zeros((n_samples, n_frames, self.n_objects, 4), dtype=np.float)
        shifted_states = np.zeros((n_samples, n_frames, self.n_objects, 4), dtype=np.float)

        for n in range(n_samples):
            np.random.seed(seed+n)
            xpos = np.random.rand(self.n_objects,1)*90 + 20
            ypos = np.random.rand(self.n_objects,1)*90 + 20
            xvel = np.zeros((self.n_objects,1))
            yvel = np.zeros((self.n_objects,1))
            states[n,0] = np.concatenate([xpos, ypos, xvel, yvel], axis=-1)
            m = [5+i for i in range(self.n_objects)]
            m = 10*m
            for t in range(n_frames):
                tempX = xpos
                tempY = ypos
                for i in range(self.n_objects):
                    xacc = 0.0
                    yacc = 0.0
                    for j in range(self.n_objects):
                        d_mag = np.linalg.norm([tempX[i] - tempX[j], tempY[i] - tempY[j]])
                        epsilon = 10000
                        k = i+j
                        xacc -= k*(tempX[i] - tempX[j]) - epsilon*m[i]*m[j]*(tempX[i] - tempX[j])/(10 + d_mag)**3
                        yacc -= k*(tempY[i] - tempY[j]) - epsilon*m[i]*m[j]*(tempY[i] - tempY[j])/(10 + d_mag)**3
                    xacc /= m[i]
                    yacc /= m[i]
                    xvel[i] += xacc*dt
                    yvel[i] += yacc*dt    
                xpos += xvel*dt
                ypos += yvel*dt

                shifted_states[n,t] =  np.concatenate([xpos, ypos, xvel, yvel], axis=-1)   
            states[n,1:,] = shifted_states[n,:-1,]
        return states, shifted_states    
    
    def visualize(self, pos, t, savepath='results_imgs/'):
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for i in range(len(pos)):
            plt.plot(pos[i,0], pos[i,1], marker='o', c=colors[i%6], ms=15+3*i)
        plt.axis([0, 128, 0, 128])
        plt.axis('off')
        plt.savefig(savepath+'%i_animate.png' % (t + 1))
        plt.close()

    def plot_trajectory(self, pos, savepath='results_imgs/'):
        colors = [mpl.cm.Oranges(np.arange(56,256)), mpl.cm.Blues(np.arange(56,256)), mpl.cm.Purples(np.arange(56,256)), mpl.cm.Greys(np.arange(56,256)),
                  mpl.cm.Greens(np.arange(56,256)), mpl.cm.Reds(np.arange(56,256))]
        for i in range(pos.shape[1]):
            plt.scatter(pos[:200,i,0], pos[:200,i,1], marker='o', c=colors[i % 6], s=5+10*i)
        plt.axis('off')
        plt.savefig(savepath)
        plt.close()
