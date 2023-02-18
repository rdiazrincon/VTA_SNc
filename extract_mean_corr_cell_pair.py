# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 09:45:11 2023

@author: Doug
"""

# Import dependencies
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# load traces
data = np.loadtxt("test_traces.csv", delimiter=",", dtype=float)

# Parameters
frames_per_second = 25
num_cells = data.shape[0]
samples = data.shape[1]
duration = samples / frames_per_second
time = np.arange(start = 0, stop=int(duration), step=1/frames_per_second)

# Plot the traces
fig, axs = plt.subplots(num_cells)
for cell_num in range(num_cells):
    axs[cell_num].plot(time,data[cell_num, 0:len(time)])


# Correlation coefficients
corrs, pvals = stats.spearmanr(np.transpose(data))

# Display the correlations
# Each square represents a "cell pair"
fig2 = plt.figure()
plt.imshow(corrs)
plt.colorbar(label="Spearman's rho")
plt.title("Adjacency Matrix")
# Fix axis so that they don't show floats and start at 1 rather than 0
plt.xlabel('Neuron #')
plt.ylabel('Neuron #')


# Extrace mean values
lower_triangle = np.tril(corrs)
mean_corr = np.mean(lower_triangle) 

