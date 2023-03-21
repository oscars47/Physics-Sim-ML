# file to generate the linear and nonlinear feature vectors for use in the nvar
# @ oscars47

## other things to implement ##
# - finding lyapunov time
# - finding fixed points
# - animate solns

import numpy as np
import scipy.integrate as sci

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
import pandas as pd

## define the 1st order differential equation system ##
def lorentz(s, t, params=(10, 28, 8/3)):
    x,y,z= s[0], s[1], s[2]
    sigma, rho, beta = params[0], params[1], params[2]
    return np.array([
        sigma * (y-x),
        x*(rho-z)-y,
        x*y - beta*z
    ])

## define integration params, solve ##
t_max = 100 # max time to integrate out to
t_min= 0
dt=0.01 # set time step
T = np.arange(t_min, t_max, .01)

# random initial condition
s0 = [0.3, 0.2, 0.5]

soln = list(zip(*sci.odeint(lorentz, s0, T, args=())))
x,y,z=soln[0], soln[1], soln[2]

# df = pd.DataFrame()
# df['x'] = x
# df['y']= y
# df['z'] = z

ax = plt.figure().add_subplot(projection='3d')
ax.plot(x, y, z)
plt.show()

# export soln here
soln = np.array(soln)
np.save('lorentz_soln.npy', soln)

## animation: this code doesn't work in 3d ##
def do_animation(soln):
    ax = plt.figure().add_subplot(projection='3d')
    # initializing a line variable
    line, = ax.plot([], [], [], lw = 3) 
    def init(): 
        line.set_data([], [], [])
        return line,
    def animate(i):
        line.set_data(soln[i][0], soln[i][1], soln[i][2])

        return line,

    # plt.show()

    anim = FuncAnimation(ax, animate, init_func = init,
                     frames = 200, interval = 20, blit = True)
  
   
    anim.save('lorentz.mp4', 
            writer = 'ffmpeg', fps = 30)
# do_animation(soln)