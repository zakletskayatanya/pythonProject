import numpy as np


def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def single_linkage(cluster1, cluster2):
    return min(euclidean_distance(point1, point2) for point1 in cluster1 for point2 in cluster2)


def agglomerative_clustering(contours):
    x, y = np.nonzero(contours == 255)

    clusters = np.column_stack((x, y))
    clusters = [clusters[i] for i in range(0, len(clusters), 10)]
    min_dist = 0
    # clusters = [[point] for point in data]

    while min_dist < 100:
        # print(clusters)
        distances = [(i, j, single_linkage(clusters[i], clusters[j]))
                     for i in range(len(clusters)) for j in range(i + 1, len(clusters))]
        i, j, min_dist = min(distances)
        # print(min_dist)

        clusters[i] = np.row_stack((clusters[i], clusters[j]))
        del clusters[j]

    return clusters
