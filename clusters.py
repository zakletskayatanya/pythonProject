import numpy as np
import scipy


def distance(point, centers, clusters):
    dist = np.linalg.norm(point, centers)
    print(dist)
    min_dist = min(dist)
    clusters[np.where(dist == min_dist)].append(point)
    return clusters

def k_means(contours):
    if contours.shape[0] < 0:
        return None
    x, y = np.nonzero(contours)

    points = np.column_stack((x[::30], y[::30]))
    clusters = {}
    k = 1000
    random_centers = points[np.random.choice(points.shape[0], k)]

    distance(points, random_centers, clusters)
