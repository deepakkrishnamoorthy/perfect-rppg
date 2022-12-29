# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 18:53:03 2022

@author: deepu
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# define a function that takes in an array of 3D points and returns
# a list of indices of the nearest neighbors for each point
def nearest_neighbors(points):
    # create a NearestNeighbors object with 1 neighbor
    nbrs = NearestNeighbors(n_neighbors=1)
    # fit the object to the points
    nbrs.fit(points)
    # return the indices of the nearest neighbors
    return nbrs.kneighbors(points, return_distance=False)



# define a function that takes in an array of 3D points and a list of indices
# of nearest neighbors and plots the points and their connections
def plot_points_and_connections(points, nn_indices):
    # create a figure and a 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')



   # plot the points
    ax.scatter(points[:,0], points[:,1], points[:,2])



   # plot the connections
    for i, nn_index in enumerate(nn_indices):
        x = [points[i][0], points[nn_index[0]][0]]
        y = [points[i][1], points[nn_index[0]][1]]
        z = [points[i][2], points[nn_index[0]][2]]
        ax.plot(x, y, z)



   # show the plot
    plt.show()



# generate some random 3D points
points = np.random.rand(10, 3)



# find the nearest neighbors for each point
nn_indices = nearest_neighbors(points)



# plot the points and their connections
plot_points_and_connections(points, nn_indices)