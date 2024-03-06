import numpy as np
from itertools import groupby


def find_objects(contours):
    x, y = np.where(contours == 255)
    points = np.column_stack((x, y))
    mask = np.zeros_like(contours)
    obj_count = 0
    height, width = contours.shape
    objects = {}

    def neighbours(i, j, object):
        stack = [(i, j)]
        while stack:
            ii, jj = stack.pop()
            if ii < 0 or jj < 0 or ii >= height or jj >= width or mask[ii, jj] != 0 or contours[ii, jj] == 0:
                continue
            mask[ii, jj] = object
            objects[object].append([ii, jj])
            stack.extend([(ii + 1, jj), (ii - 1, jj), (ii, jj + 1), (ii, jj - 1), (ii - 1, jj - 1), (ii + 1, jj - 1),
                          (ii - 1, jj + 1), (ii + 1, jj + 1)])

    for point in points:
        if mask[point[0], point[1]] == 0:
            obj_count += 1
            objects[obj_count] = []
            # mask[point[0], point[1]] = obj_count
            # objects[obj_count].append([point[0], point[1]])
            neighbours(point[0], point[1], obj_count)

    return objects
