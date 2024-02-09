import numpy as np
import scipy


def min_distance(X, cluster1, cluster2):
    min_dist = np.inf
    for i in cluster1:
        for j in cluster2:
            dist = np.linalg.norm(X[i] - X[j])
            if dist < min_dist:
                min_dist = dist
    return min_dist


def find_clusters(contours):
    x, y = np.nonzero(contours == 255)  # координаты контуров

    points = np.column_stack((x, y))
    print(len(points))

    step = 50
    clusters = [[i] for i in range(0, len(points), step)]

    if 0 < len(points) < 15000:

        while True:

            min_i, min_j = None, None
            min_dist = float('inf')
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = min_distance(points, clusters[i], clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        min_i, min_j = i, j

            if min_dist > 50:
                break

            clusters[min_i] = clusters[min_i] + clusters[min_j]
            del clusters[min_j]

    return clusters, points


def predict(cont):
    clustered_points = []
    clusters, X = find_clusters(cont)
    for cluster in clusters:
        min_point = np.min(X[cluster], axis=0)
        max_point = np.max(X[cluster], axis=0)
        clustered_points.append((min_point, max_point))
    return clustered_points
