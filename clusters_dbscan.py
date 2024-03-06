import numpy as np


def dbscan_naive(contours, eps, m):

    if contours.shape[0] < 0:
        return None
    x, y = np.nonzero(contours)
    points = np.column_stack((x[::15], y[::15]))
    C = 0

    visited_points = [[-1, -1]]
    clustered_points = []
    clusters = {C: []}
    dist = np.linalg.norm(points[:, None] - points, axis=2)
    dist = np.where(dist == 0, float('inf'), dist)

    def expand_cluster(p, neighbours):
        if C not in clusters:
            clusters[C] = []
        clusters[C].append(p)
        clustered_points.append(p)
        # clusters[C].extend(neighbours)
        # clustered_points.extend(neighbours)
        while len(neighbours) > 0:
            q = neighbours[len(neighbours) - 1]
            neighbours = np.delete(neighbours, -1, 0)
            if q in np.array(visited_points):
                continue
            visited_points.append(q)
            # q_indx = np.where(np.all(q == points, axis=1))[0]
            neighbourz_indx = np.where(dist[np.where(np.all(q == points, axis=1))[0]] < eps)[1]
            neighbourz = points[neighbourz_indx]
            if len(neighbourz) >= m:
                neighbours = np.append(neighbours, neighbourz, 0)
                if not q in np.array(clustered_points):
                    clustered_points.append(q)
                    clusters[C].append(q)
                    # clusters[C].extend(neighbourz)
                    # clustered_points.extend(neighbourz)

    for i, p in enumerate(points):
        if p in np.array(visited_points):
            continue
        visited_points.append(p)
        # neighbours_indx = np.where(dist[i] < eps)[0]
        neighbours = points[np.where(dist[i] < eps)[0]]
        if len(neighbours) >= m:
            expand_cluster(p, neighbours)
            C += 1

    return clusters
