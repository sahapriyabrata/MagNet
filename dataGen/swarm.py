import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class swarm:
    def __init__(self, n_predator=1,  n_prey=50):
        self.n_objects = n_predator + n_prey
        self.n_predator = n_predator
        self.n_prey = n_prey
        
    def generate_data(self, n_samples=1, n_frames=1000, dt=0.01, seed=0):
        states = np.zeros((n_samples, n_frames, self.n_objects, 4), dtype=np.float)
        shifted_states = np.zeros((n_samples, n_frames, self.n_objects, 4), dtype=np.float)

        for n in range(n_samples):    
            np.random.seed(seed+n)
            xpos = np.random.rand(self.n_prey,1)*90 + 20
            ypos = np.random.rand(self.n_prey,1)*90 + 20
            xpos = np.concatenate([xpos, np.random.rand(self.n_predator,1)*40 + 40], axis=0)
            ypos = np.concatenate([ypos, np.random.rand(self.n_predator,1)*40 + 40], axis=0)
            xvel = np.zeros((self.n_objects,1))
            yvel = np.zeros((self.n_objects,1))
            states[n,0] = np.concatenate([xpos, ypos, xvel, yvel], axis=-1)
            a = 1
            b = 5e2
            c = 5e2
            d = 1e4
            for t in range(n_frames):
                tempX = xpos
                tempY = ypos
                for i in range(self.n_prey):
                    xacc = 0.0
                    yacc = 0.0
                    for j in range(self.n_prey):
                        d_mag = np.linalg.norm([tempX[i] - tempX[j], tempY[i] - tempY[j]])
                        xacc -= a*(tempX[i] - tempX[j]) - c*(tempX[i] - tempX[j])/(1 + d_mag)**2
                        yacc -= a*(tempY[i] - tempY[j]) - c*(tempY[i] - tempY[j])/(1 + d_mag)**2
                    xacc /= self.n_prey
                    yacc /= self.n_prey
                    for k in range(self.n_predator):    
                        d_mag = np.linalg.norm([tempX[i] - tempX[self.n_prey+k], tempY[i] - tempY[self.n_prey+k]])
                        xacc += b*(tempX[i] - tempX[self.n_prey+k])/(10 + d_mag)**2
                        yacc += b*(tempY[i] - tempY[self.n_prey+k])/(10 + d_mag)**2
                    xvel[i] = xacc
                    yvel[i] = yacc                       
                for i in range(self.n_predator):   
                    xacc = 0.
                    yacc = 0.
                    for j in range(self.n_prey):
                        d_mag = np.linalg.norm([tempX[j] - tempX[self.n_prey+i], tempY[j] - tempY[self.n_prey+i]])
                        xacc += c*(tempX[j] - tempX[self.n_prey+i])/(10 + d_mag)**2
                        yacc += c*(tempY[j] - tempY[self.n_prey+i])/(10 + d_mag)**2
                    xacc /= self.n_prey
                    yacc /= self.n_prey
                    xvel[self.n_prey+i] = xacc
                    yvel[self.n_prey+i] = yacc  
                xpos += xvel*dt
                ypos += yvel*dt

                shifted_states[n, t] = np.concatenate([xpos, ypos, xvel, yvel], axis=-1)    
            states[n,1:,] = shifted_states[n,:-1,]
            
        return states, shifted_states                

    def visualize(self, states, i, savepath='results_imgs/'):
        ax = plt.gca()
        plt.plot(states[self.n_prey:,0], states[self.n_prey:,1], 'o', color=(0.75, 0, 0), ms = 15)
        for n in range(self.n_prey):
            poly = np.array([[states[n,0], states[n,1]], [states[n,0]-2, states[n,1]], [states[n,0]+2, states[n, 1]+2], [states[n,0], states[n, 1]-2]])
            prey = plt.Polygon(poly, facecolor='k')
            direction = np.angle(states[n,2] + states[n,3] * 1j, deg=True) - 45
            rot = mpl.transforms.Affine2D().rotate_deg_around(states[n,0], states[n,1], direction) + ax.transData
            prey.set_transform(rot)
            ax.add_patch(prey)
        #plt.axis([25, 115, 25, 115])
        plt.axis([0, 128, 0, 128])
        plt.axis('off')
        plt.savefig(savepath+'%i_animate.png' % (i + 1))
        plt.close()
        
    def plot_trajectory(self, pos, savepath='results_imgs/'):
        colors = [mpl.cm.Oranges(np.arange(56,256)), mpl.cm.Blues(np.arange(56,256)), mpl.cm.Purples(np.arange(56,256)), mpl.cm.Greys(np.arange(56,256)),
                  mpl.cm.Greens(np.arange(56,256)), mpl.cm.Reds(np.arange(56,256))]
        for i in range(self.n_prey):
            plt.scatter(pos[:,i,0], pos[:,i,1], marker='o', c=colors[1], s=5)
        for i in range(self.n_predator):    
            plt.scatter(pos[:,self.n_prey+i,0], pos[:,self.n_prey+i,1], marker='o', c=colors[-1], s=50)
        plt.axis('off')
        plt.savefig(savepath)
        plt.close()
               
