__author__ = 'raza'

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('simulation_data.txt', skiprows=1, delimiter=',')

# number of values of r, from [0.0, 1.0] with a difference of 0.01
num_r = 101
# values of percolation threshold against each r
pct = np.zeros(num_r)
# total number of network configurations per r
num_nets = 123
# total number of stochastic simulations for each setting
sims_per_r = 100
# total number of simulations
total_points = num_r * num_nets * sims_per_r
# cutoff threshold for the non-zero conductance values
thresh = 35
for i in xrange(num_r):
    j = i * num_nets * sims_per_r
    while j < total_points:
        if np.count_nonzero(data[j: j + sims_per_r, -1]) >= thresh:
            pct[i] = data[j, 0]
            break
        j += sims_per_r

area = 1600
# normalizing to get packing density
print pct
pct /= area
print pct


# plotting the data here
x = np.arange(0.0, 1.01, 0.01)
plt.plot(x, pct, c='g', label='PCT')
plt.xlabel('r')
plt.ylabel('G')
plt.title('Percolation Thresholds')
plt.legend()
plt.show()
