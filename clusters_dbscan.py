import numpy as np


def dbscan_naive(contours, eps, m):
    x, y = np.nonzero(contours)
    if len(x) < 1 or len(y) < 1:
        return None
    xx = x[::30]
    yy = y[::30]
    points = np.column_stack((xx, yy))
    # print(points)
    # p_ = np.array(points)
    # print(type(points))
    NOISE = -10
    C = -1

    visited_points = []
    clustered_points = []
    clusters = {NOISE: []}
    dist = np.linalg.norm(points[:, None] - points, axis=2)
    dist = np.where(dist == 0, float('inf'), dist)

    def expand_cluster(p, neighbours):
        if C not in clusters:
            clusters[C] = []
        clusters[C].append(p)
        clustered_points.append(p)
        while len(neighbours) > 0:
            q = neighbours[len(neighbours) - 1]
            # ppp=[np.array([20, 451]),np.array([280, 1])]
            # print(ppp)
            neighbours = np.delete(neighbours, -1, 0)
            # oo = np.all(q == ppp, axis=0)
            # print(oo.any())
            # print((q == visited_points))
            # print("ds")
            # print(np.all(q == visited_points, axis=1))
            # print(np.all(q == visited_points, axis=1).any())
            if not np.all(q == visited_points, axis=1).any():
                visited_points.append(q)
                q_indx = np.where((points[:, 0] == q[0]) & (points[:, 1] == q[1]))[0]
                neighbourz_indx = np.where(dist[q_indx] < eps)[1]
                neighbourz = points[neighbourz_indx]
                if len(neighbourz) > m:
                    neighbours = np.append(neighbours, neighbourz, 0)
            if not np.all(q == clustered_points, axis=1).any():
                clustered_points.append(q)
                clusters[C].append(q)
                # if np.all(q == clusters[NOISE], axis=1).any():
                #     print(q)
                #     print(clusters[NOISE])
                #     (clusters[NOISE].remove(q)).any()

    for i, p in enumerate(points):
        if len(visited_points) > 0 and (p == visited_points).any():
            continue
        visited_points.append(p)
        neighbours_indx = np.where(dist[i] < eps)[0]
        neighbours = points[neighbours_indx]
        if len(neighbours) < m:
            clusters[NOISE].append(p)
        else:
            C += 1
            expand_cluster(p, neighbours)

    return clusters
