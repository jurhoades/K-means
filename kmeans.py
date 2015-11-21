# Imports
from copy import deepcopy
from math import sqrt
from random import randrange
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Constants
NUM_ITERATIONS = 100

# Functions -------------------------------------------------------------------
def find_cluster(cluster_list, data_pt):
    '''Assigns data point to nearest cluster'''
    cluster_dist_list = []
    for cluster in range(len(cluster_list)):
        cluster_dist = 0
        for dim in range(len(cluster_list[cluster])):
            cluster_dist += (data_pt[dim] - cluster_list[cluster][dim])**2
        cluster_dist_list.append(cluster_dist)

    return cluster_dist_list.index(min(cluster_dist_list))

# Main ------------------------------------------------------------------------
def main():
    if len(sys.argv) != 3:
        print('Usage: kmeans.py {/path/to/data_file} {# of clusters}')
        sys.exit(1)
    data_file = sys.argv[1]
    K = int(sys.argv[2])

    #read data into data frame
    data_pts = pd.read_table(data_file, sep='\t', header=None)
    num_features = len(data_pts.columns)  #if using real data
    # num_features = len(data_pts.columns) - 1  #if using generate-clusters.py data 

    #initialize clusters at randomly selected points
    clust_cents = []
    for i in range(K):
        clust = list(data_pts.ix[randrange(len(data_pts)),:num_features-1])
        clust_cents.append(clust)

    #Main Iteration
    for iteration in range(NUM_ITERATIONS):
        #assign data points to closest cluster (euclidean distance)
        for data_pt in range(len(data_pts)):
            cluster_ix = find_cluster(clust_cents, data_pts.ix[data_pt])
            data_pts.ix[data_pt,"Cluster"] = cluster_ix

        #move cluster centers to centroids
        old_clust_cents = deepcopy(clust_cents)
        for cluster in range(len(clust_cents)):
            cluster_data = data_pts[data_pts["Cluster"] == cluster]
            for dim in range(num_features):
                clust_cents[cluster][dim] = cluster_data[dim].mean()

        #breaks iteration if clusters don't move
        if old_clust_cents == clust_cents:
            print("Number of iterations:", iteration)
            break

    #print cluster features
    print("Cluster Locations:")
    for cluster in clust_cents:
        print(cluster)

    print("Cluster Sizes:")
    cluster_sizes = []
    for cluster in range(len(clust_cents)):
        cluster_sizes.append(len(data_pts[data_pts["Cluster"] == cluster]))
        print("Cluster #{0}:".format(cluster+1), len(data_pts[data_pts["Cluster"] == cluster]))

    if K <= 6 and num_features == 2:
        #plot data points and centroids
        #only works for 2 features, max K=6
        colors = ["Red","Purple","LightGreen", "LightBlue", "Yellow", "Cyan"]
        markers = ["^", "*", "o", "s", "p", "x"]
        for data_pt in range(len(data_pts)):
            color = colors[int(data_pts.ix[data_pt,"Cluster"])]
            marker = markers[int(data_pts.ix[data_pt,2])]
            plt.scatter(data_pts.ix[data_pt,0],data_pts.ix[data_pt,1], s=50, c=color, marker=marker)
        for i in range(K):
            plt.scatter(clust_cents[i][0],clust_cents[i][1], s=100, c="Orange")
        plt.show()


if __name__ == '__main__':
    main()