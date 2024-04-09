import math

import cv2
import numpy as np
import gaussian_filter_no_opencv as gauss
from scipy.signal import convolve2d


def func_norm(i, j):
    sigma = 2
    return 1 / np.sqrt(2 * np.pi)/sigma * np.exp(-1 / 2 * ((i - j)**2)/sigma**2)


class OpticalFlowProcessor:

    def __init__(self, frame1):
        self.history_points = []
        self.color = (0, 0, 255)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def calculate_optical_flow(self, frame1, frame2, treck, gradient_x, gradient_y, gradient_time, mask):

        w_size = 8

        u = np.zeros(gradient_x.shape)
        v = np.zeros(gradient_x.shape)
        ugol = np.zeros(gradient_x.shape)
        func_gauss = list(map(func_norm, range(2*w_size), range(2*w_size)))
        weight = np.array(list(func_gauss) * 2 * w_size)
        for rect in treck:
            y, x, h, w = rect
            for i in range(x+w_size, x+w - w_size):
                for j in range(y+w_size, y+h - w_size):
                    Ix1 = gradient_x[i - w_size:i + w_size, j - w_size:j + w_size].flatten()
                    Iy1 = gradient_y[i - w_size:i + w_size, j - w_size:j + w_size].flatten()
                    It = gradient_time[i - w_size:i + w_size, j - w_size:j + w_size].flatten()

                    M = np.array([[np.sum((weight*Ix1) ** 2), np.sum(weight*Ix1 * Iy1)],
                                  [np.sum(weight*Iy1 * Ix1), np.sum((weight*Iy1) ** 2)]])
                    b = -np.array([np.sum(Ix1 * It), np.sum(Iy1 * It)])
                    if np.linalg.det(M) <= 0:
                        continue
                    uv = np.linalg.solve(M, b)
                    u[i, j] = uv[0]
                    v[i, j] = uv[1]
                    k = 0.15
                    R = np.linalg.det(M)-k*np.trace(M)**2
                    if R > k:
                        ugol[i, j] = R

        ug = np.column_stack(np.where(ugol > 0.01*ugol.max()))

        for p in ug:
            j, i = p
            # print(u[j,i], v[j,i])
            mask = cv2.line(mask, (i, j), (i + int(u[j, i]), j + int(v[j, i])), (0, 255, 0), 2)
            frame1 = cv2.circle(frame1, (i, j), 1, (255, 0, 0), 1)
        img = cv2.add(frame1, mask)

        return img
