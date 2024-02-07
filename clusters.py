import numpy as np
import scipy
import math


def euclid_dist(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def linkage(clust1, clust2):
    return min(euclid_dist(p1, p2) for p1 in clust1 for p2 in clust2)


def find_clusters(contours):
    x, y = np.nonzero(contours == 255)  # координаты контуров

    # clusters_x = np.zeros([1, len(x) // 20])
    # clusters_y = np.zeros([1, len(y) // 20])
    # print(clusters_y.shape,  type(clusters_y))

    clusters_x = [[x[i]] for i in range(0, len(x), 70)]  # инициализация кластеров
    clusters_y = [[y[i]] for i in range(0, len(y), 70)]
    # clusters = np.column_stack((clusters_x, clusters_y))
    # clusters = clusters.tolist()

    if len(clusters_x) > 0 and len(clusters_x) > 0:

        # while len(clusters) > 2:
        #
        #     distances = [(i, j, linkage(clusters[i], clusters[j]))
        #                  for i in range(len(clusters)) for j in range(i + 1, len(clusters))]
        #     i, j, min_dist = min(distances)
        #
        #     if min_dist >= 20:
        #         break
        #
        #     clusters[i] = clusters[i] + clusters[j]
        #     del clusters[j]

        # print(len(clusters), clusters)
        # for i in range(len(clusters)):
        #     clust_x[i] = [clusters[i][j] for j in range(0, len(clusters[i]), 2)]
        #     clust_y[i] = [clusters[i][j] for j in range(1, len(clusters[i]), 2)]

        while True:

            clusters_x_mean = np.array(list(map(np.mean, clusters_x)))
            clusters_y_mean = np.array(list(map(np.mean, clusters_y)))

            column_stack_clusters = np.column_stack((clusters_x_mean, clusters_y_mean))
            distance = scipy.spatial.distance.cdist(column_stack_clusters, column_stack_clusters, 'euclidean')

            np.fill_diagonal(distance, np.inf)
            min_distance_index = np.argmin(distance)
            min_distance = distance.min()

            if min_distance >= 150:
                break

            i, j = np.unravel_index(min_distance_index, distance.shape)
            clusters_x[i] = np.concatenate((clusters_x[i], clusters_x[j]))
            clusters_y[i] = np.concatenate((clusters_y[i], clusters_y[j]))
            del clusters_y[j]
            del clusters_x[j]
        clusters = np.column_stack((clusters_x, clusters_y))
        clusters = clusters.tolist()

    return clusters
