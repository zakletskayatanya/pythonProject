import numpy as np
import scipy


def find_clusters(contours):
    x, y = np.nonzero(contours == 255)  # координаты контуров

    clusters_x = [[x[i]] for i in range(0, len(x), 20)]  # инициализация кластеров
    clusters_y = [[y[i]] for i in range(0, len(y), 20)]

    if len(clusters_x) > 0 and len(clusters_x) > 0:

        while True:

            clusters_x_mean = np.array(list(map(np.mean, clusters_x)))
            clusters_y_mean = np.array(list(map(np.mean, clusters_y)))

            column_stack_clusters = np.column_stack((clusters_x_mean, clusters_y_mean))

            distance = scipy.spatial.distance.cdist(column_stack_clusters, column_stack_clusters)

            np.fill_diagonal(distance, np.inf)
            min_distance_index = np.argmin(distance)
            min_distance = distance.min()

            if min_distance >= 80:
                break

            i, j = np.unravel_index(min_distance_index, distance.shape)
            clusters_x[i] = np.concatenate((clusters_x[i], clusters_x[j]))
            clusters_y[i] = np.concatenate((clusters_y[i], clusters_y[j]))
            del clusters_y[j]
            del clusters_x[j]

    return clusters_x, clusters_y
