
import numpy as np
from scipy.spatial.distance import cdist


def distance(cluster1, cluster2):
    return cdist(cluster1, cluster2)


def min_distance(cluster1, cluster2):
    return (distance(cluster1, cluster2))


def find_clusters(contours):
    if contours.shape[0] < 0:
        return None
    y, x = np.nonzero(contours)
    points = np.column_stack((x, y))
    print(len(points))

    clusters = [[i] for i in points[::150]]
    min_dist = 0
    ii = 0
    jj = 1
    count=0


    while True:

        if jj >= len(clusters):
            ii += 1
            jj = ii+1
        if ii >= len(clusters)-1:
            return clusters

        cluster1 = clusters[ii]
        cluster2 = clusters[jj]
        if np.min(cdist(cluster1, cluster2)) < 5:
            cluster1.extend(cluster2)
            clusters.pop(jj)
        else:
            jj += 1



    #     P = np.zeros((len(clusters), len(clusters)))+float('inf')
    #
    #     for i, cluster1 in enumerate(clusters, start=len(clusters)):
    #         for j, cluster2 in enumerate(clusters[i + 1:], start=i + 1):
    #             P[i, j] = np.min(cdist(cluster1, cluster2))
    #
    #
    #     min_dist = np.min(P)
    #
    #     min_index_j, min_index_i = np.where(P == min_dist)
    #     pop_index = sorted(set(min_index_j), reverse=True)
    #
    #     for i in reversed(range(0, len(min_index_i))):
    #         clusters[min_index_i[i]].extend(clusters[min_index_j[i]])
    #
    #     for i in pop_index:
    #         clusters.pop(i)
    #     print('1')
    #
    # return clusters