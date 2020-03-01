import numpy as np
import pymunk
import matplotlib as mpl
import matplotlib.pyplot as plt

class bouncing_balls(object):

      def __init__(self, n_objects=5):
          self.n_objects = n_objects
          self.masses = 10 * np.ones(n_objects)
          self.radii = 10 * np.ones(n_objects)

      def create_env(self, seed=0):

          self.space = pymunk.Space()
          self.space.gravity = (0.0, 0.0)
          self.objects = []

          np.random.seed(seed)
          xpos = np.random.rand(self.n_objects,1)*90 + 20
          ypos = np.random.rand(self.n_objects,1)*90 + 20
          xvel = np.random.rand(self.n_objects,1)*400 - 200
          yvel = np.random.rand(self.n_objects,1)*400 - 200
              
          for i in range(self.n_objects):
              mass = self.masses[i]
              radius = self.radii[i]
              moment = pymunk.moment_for_circle(mass, 0, radius, (0,0))
              body = pymunk.Body(mass, moment)
              body.position = (xpos[i], ypos[i])
              body.velocity = (xvel[i], yvel[i])
              shape = pymunk.Circle(body, radius, (0,0))
              shape.elasticity = 1.0
              self.space.add(body, shape)
              self.objects.append(body)

          static_body = self.space.static_body
          static_lines = [pymunk.Segment(static_body, (0.0, 0.0), (0.0, 128.0), 0),
                          pymunk.Segment(static_body, (0.0, 0.0), (128.0, 0.0), 0), 
                          pymunk.Segment(static_body, (128.0, 0.0), (128.0, 128.0), 0),
                          pymunk.Segment(static_body, (0.0, 128.0), (128.0, 128.0), 0)]

          for line in static_lines:
              line.elasticity = 1.
          self.space.add(static_lines)

      def get_state(self):
          state = np.zeros((self.n_objects, 4))
          for i in range(self.n_objects):
              state[i, :2] = np.array([self.objects[i].position[0], self.objects[i].position[1]])
              state[i, 2:] = np.array([self.objects[i].velocity[0], self.objects[i].velocity[1]])
              
          return state

      def step(self, dt=0.01):
          self.space.step(dt)
          return self.space

      def generate_data(self, n_samples=1, n_frames=1000, dt=0.01, seed=0):
          states = np.zeros((n_samples, n_frames, self.n_objects, 4))      
          
          for n in range(n_samples):
              self.create_env(seed=seed+n)
              for t in range(n_frames):
                  states[n,t] = self.get_state()
                  if t > 0:
                      states[n,t,:,2:] = (states[n,t,:,:2] - states[n,t-1,:,:2])/dt
                  self.step(dt=dt)
          
          inputs = []
          targets = []
          for i in range(n_samples):
              for j in range(n_frames-1):
                  if np.array_equal(states[i,j,:,2:].astype(int), states[i, j+1,:,2:].astype(int)):
                      continue
                  inputs.append(states[i,j])
                  targets.append(states[i,j+1])
          inputs = np.array(inputs)
          targets = np.array(targets)

          return states, inputs, targets

      def visualize(self, pos, t, savepath='results_imgs/'):
          colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']    
          fig, ax = plt.subplots(figsize=(6,6))
          box = plt.Rectangle((0,0), 128, 128, linewidth=5, edgecolor='k', facecolor='none')
          ax.add_patch(box)
          for i in range(len(pos)):
              circle = plt.Circle((pos[i,0], pos[i,1]), radius=self.radii[i], color=colors[i%6])
              ax.add_patch(circle)
          plt.axis([0, 128, 0, 128])
          plt.axis('off')
          plt.savefig(savepath+'%i_animate.png' % (t + 1))
          plt.close()

      def plot_trajectory(self, pos, savepath='results_imgs/traj.png'):
          colors = [mpl.cm.Oranges(np.arange(56,256)), mpl.cm.Blues(np.arange(56,256)), mpl.cm.Purples(np.arange(56,256)), mpl.cm.Greys(np.arange(56,256)),
                    mpl.cm.Greens(np.arange(56,256)), mpl.cm.Reds(np.arange(56,256))]
          for i in range(pos.shape[1]):
              plt.scatter(pos[:50,i,0], pos[:50,i,1], marker='o', c=colors[i % 6][::4], s=50)
          plt.axis('off')
          plt.savefig(savepath)
          plt.close()


