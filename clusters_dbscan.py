import numpy as np


def distance(p, q):
    return np.linalg.norm(p - q)


def dbscan_naive(contours, eps, m):
    x, y = np.nonzero(contours == 255)  # координаты контуров
    if len(x) < 1 or len(y) < 1 or len(x) > 10000 or len(y) > 10000:
        return None

    points = np.column_stack((x, y))

    P = [points[i] for i in range(0, len(points), 50)]
    NOISE = 0
    C = 0

    visited_points = []
    clustered_points = []
    clusters = {NOISE: []}

    def region_query(p):
        return [q for q in P if distance(p, q) < eps]

    def expand_cluster(p, neighbours):
        if C not in clusters:
            clusters[C] = []
        clusters[C].append(p)
        clustered_points.append(p)
        while neighbours:
            q = neighbours.pop()
            if not np.array_equal(q, visited_points):
                visited_points.append(q)
                neighbourz = region_query(q)
                if len(neighbourz) > m:
                    neighbours.extend(neighbourz)
            if not np.array_equal(q, clustered_points):
                clustered_points.append(q)
                clusters[C].append(q)
                if np.array_equal(q, clusters[NOISE]):
                    clusters[NOISE].remove(q)

    for p in P:
        if np.array_equal(p, visited_points):
            continue
        visited_points.append(p)
        neighbours = region_query(p)
        if len(neighbours) < m:
            clusters[NOISE].append(p)
        else:
            C += 1
            expand_cluster(p, neighbours)

    return clusters
