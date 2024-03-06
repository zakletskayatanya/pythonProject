import numpy as np
from scipy.spatial.distance import cdist


def distance(cluster1, cluster2):
    return cdist(cluster1, cluster2)


def min_distance(cluster1, cluster2):
    return np.min(distance(cluster1, cluster2))


def find_clusters(contours):
    if contours.shape[0] < 0:
        return None
    x, y = np.nonzero(contours)
    points = np.column_stack((x[::20], y[::20]))

    clusters = [[i] for i in points]
    min_dist = 0

    while min_dist < 100:

        P = np.zeros((len(clusters), len(clusters)))

        for i in range(len(clusters)):
            cluster1 = np.array(clusters[i])
            for j in range(i + 1, len(clusters)):
                cluster2 = np.array(clusters[j])
                P[i, j] = min_distance(cluster1, cluster2)

        P = np.where(P == 0, float('inf'), P)

        min_dist = np.min(P)

        min_index_i, min_index_j = np.where(P == min_dist)
        min_index = np.column_stack((min_index_i, min_index_j))
        min_index = sorted(min_index, key=lambda k:[k[0], k[1]])

        for i in range(len(min_index) - 1, 0, -1):
            clusters[min_index[i][0]].extend(clusters[min_index[i][1]])

        for i in sorted(set(min_index_j), reverse=True):
            clusters.pop(i)

    return clusters
