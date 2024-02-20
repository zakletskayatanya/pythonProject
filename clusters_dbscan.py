import numpy as np


def distance(p, q):
    return np.linalg.norm(p - q)


def dbscan_naive(contours, eps, m):
    x, y = np.nonzero(contours == 255)  # координаты контуров
    if len(x) < 1 or len(y) < 1 or len(x) > 10000 or len(y) > 10000:
        return None

    points = np.column_stack((x, y))

    P = [points[i] for i in range(0, len(points), 50)]
    NOISE = -10
    C = -1

    visited_points = []
    clustered_points = []
    clusters = {NOISE: []}
    dist = np.linalg.norm(points[:, None] - points, axis=2)

    def region_query(p):
        return [q for q in P if distance(p, q) < eps]

    def expand_cluster(p, neighbours):
        if C not in clusters:
            clusters[C] = []
        clusters[C].append(p)
        clustered_points.append(p)
        while len(neighbours)>0:
            q = neighbours.pop()
            if not np.array_equal(q, visited_points):
                visited_points.append(q)
                q_indx = np.where(points == q)
                print(q_indx)
                print("22")
                neighbourz_indx = np.where(dist[q_indx] < eps)[0]
                neighbourz = points[neighbourz_indx]
                if len(neighbourz) > m:
                    neighbours.extend(neighbourz)
            if not np.array_equal(q, clustered_points):
                clustered_points.append(q)
                clusters[C].append(q)
                if np.array_equal(q, clusters[NOISE]):
                    clusters[NOISE].remove(q)

    for i, p in enumerate(points):
        if np.array_equal(p, visited_points):
            continue
        visited_points.append(p)
        neighbours_indx = np.where(dist[i] < eps)[0]
        neighbours = list(points[neighbours_indx])
        print(neighbours)
        if len(neighbours) < m:
            clusters[NOISE].append(p)
        else:
            C += 1
            expand_cluster(p, neighbours)

    return clusters