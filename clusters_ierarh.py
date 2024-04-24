import numpy as np
from scipy.spatial.distance import cdist


def distance(cluster1, cluster2):
    return cdist(cluster1, cluster2)


def min_distance(cluster1, cluster2):
    return np.min(distance(cluster1, cluster2))


def find_clusters(contours):
    if contours.shape[0] < 0:
        return None
    y, x = np.nonzero(contours)
    points = np.column_stack((x, y))

    clusters = [[i] for i in points[::5]]
    min_dist = 20
    ii = 0
    jj = 1

    while True:
        if jj >= len(clusters):
            ii += 1
            jj = ii + 1
        if ii >= len(clusters) - 1:
            return clusters

        cluster1 = clusters[ii]
        cluster2 = clusters[jj]
        if min_distance(cluster1, cluster2) < min_dist:
            cluster1.extend(cluster2)
            clusters.pop(jj)
        else:
            jj += 1
