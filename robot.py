# welcome to my portal motherfuckas
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.spatial.distance import pdist
from matplotlib.animation import FuncAnimation

#make fig
fig = plt.figure()
ax = plt.gca()

#making widgets that create initial conditions
q = widgets.IntSlider(min=0, max=10)
f = widgets.IntSlider(min=0, max=1)    #coefficient of friction
display(q, f)



#make board, pockets
board = plt.Rectangle((0,0), width=10, height=15, facecolor="green")
#pockets, starting at origin going clockwise
pocket1 = plt.Circle((0,0), radius=1.5, facecolor="black")
pocket2 = plt.Circle((15,0), radius=1.5, facecolor="black")
pocket3 = plt.Circle((15,10), radius=1.5, facecolor="black")
pocket4 = plt.Circle((0,10), radius=1.5, facecolor="black")

ax.add_patch(board)
ax.add_patch(pocket1)
ax.add_patch(pocket2)
ax.add_patch(pocket3)
ax.add_patch(pocket4)

#widgets to make move
angle = widget.IntSlider(min=0, max=6.28)          #angle you hit the ball, between 0 and 2pi
speed = widget.IntSlider(min=.01, max=10)    #force you hit the ball with
display(angle, speed)

#make velocity vector plot to help player choose angle and speed, centered at cue ball
velocity = plt.Line2D((cue.x, speed*np.cos(angle)), (cue.y))
# TODO:
# outline classes
# main loop?
# what should threshhold velocities be

class Ball:
    def __init__(self, x, y, r=1, q=1):
        self.x, self.y, self.r, self.q = x, y, r, q
        self.vx, self.vy = 0, 0
    # method to detect collisions
    def collidesWith(self, obj):
        d = np.sqrt((self.x-obj.x)**2 + (self.y-obj.y)**2)
        if d <= self.r + obj.r:
            return true
        else:
            return false

# Number of balls, M
M = 6
# Array of ball objects
balls = np.empty(M, dtype=objects)
"""
# Generalized start_positions code if wanted
start_positions = []
y0 = 2
x0 = 4
pos = 0
for m in range(M):
    start_positions.append((x0, y0))
    pos += 1
    x0 += 1
    if (pos == 2):
        y0 += 1
        x0 = 4
        pos = 0
"""
# Instantiate ball objects
start_positions = [(4, 2), (5, 2), (4, 3), (5, 3), (4, 4), (5, 4)]
for i in length(M):
    balls[i] = Ball(start_positions[i][0], start_positions[i][1]. q=q)

cue = Ball(5, 10, q=0)

# (x, y) positions of each ball
X = np.array(list([ball.x, ball.y] for ball in balls))
# calculate square of pairwise (euclidean) distances btwn. all balls
d_sq = pdist(X, 'sqeuclidean')
h = .1 # Seconds
velocity_threshold = 1e-3 # m/s: if velocity <= this value, velocity ~= 0

# When user says go, run this function!
def go():
    # balls move to equilibrium positions
    # calculate cue ball motion until it hits something
    # fuck
    # solve for electrostatic forces in solve_ivp
    # handle collisions afterwards

# TODO:
# set tao (time step size)
# calculate force :) qq/r**2
# do collision stuff
# pick velocity threshold to stop calculations/consider system static (what ~= 0)
