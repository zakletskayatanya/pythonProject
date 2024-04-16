import math

import cv2
import numpy as np
import gaussian_filter_no_opencv as gauss
from scipy.signal import convolve2d


def func_norm(i, j):
    sigma = 2
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-1 / 2 * ((i - j) ** 2) / sigma ** 2)


class OpticalFlowProcessor:

    def __init__(self, frame1):
        self.history_points = []
        self.color = (0, 0, 255)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def calculate_optical_flow(self, frame11, frame1, frame2, treck, gradient_x1, gradient_y1, gradient_time, mask, frame_counter):

        w_size = 5

        U = []
        V = []

        sobel_x = np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])
        sobel_y = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])
        k = 0.06

        for iter in range(3, -1, -1):
            frame1_pyr = cv2.resize(frame1, (840 // 2 ** iter, 480 // 2 ** iter), interpolation=cv2.INTER_AREA)
            frame2_pyr = cv2.resize(frame2, (840 // 2 ** iter, 480 // 2 ** iter), interpolation=cv2.INTER_AREA)

            gradient_x = convolve2d(frame1_pyr,  sobel_x, mode='same')
            gradient_y = convolve2d(frame1_pyr,  sobel_y, mode='same')
            u = np.zeros(gradient_x.shape)
            v = np.zeros(gradient_x.shape)
            ugol = np.zeros(gradient_x.shape)
            for rect in treck:
                y, x, h, w = rect
                y = y // 2 ** iter
                x = x // 2 ** iter
                h = h // 2 ** iter
                w = w // 2 ** iter
                for i in range(x, x + w):
                    for j in range(y, y + h):
                        # print(i, j)
                        Ix1 = gradient_x[i - w_size:i + w_size+1, j - w_size:j + w_size+1].flatten()
                        Iy1 = gradient_y[i - w_size:i + w_size+1, j - w_size:j + w_size+1].flatten()
                        # print(gradient_x[i - w_size:i + w_size, j - w_size:j + w_size+1])
                        # print(gradient_x[i,j])
                        weight = func_norm(i, j)
                        M = weight * np.array([[np.sum(Ix1 ** 2), np.sum(Ix1 * Iy1)],
                                               [np.sum(Iy1 * Ix1), np.sum(Iy1 ** 2)]])
                        if np.linalg.det(M) == 0:
                            continue
                        R = np.linalg.det(M) - k * np.trace(M) ** 2
                        # if R < k:
                        #     continue

                        It = -frame2_pyr[i - w_size:i + w_size+1, j - w_size:j + w_size+1].flatten()
                        b = -weight * np.array([np.sum(Ix1 * It), np.sum(Iy1 * It)])
                        ugol[i, j] = R
                        uv = np.linalg.solve(M, b)
                        u[i, j] = uv[0]
                        v[i, j] = uv[1]

            U.append(u)
            V.append(v)
        u_t, v_t = np.zeros_like(U[0]), np.zeros_like(V[0])
        for i in range(len(U) - 1):
            u_d = cv2.resize(U[i], (U[i].shape[1] * 2, U[i].shape[0] * 2), interpolation=cv2.INTER_AREA)
            v_d = cv2.resize(V[i], (V[i].shape[1] * 2, V[i].shape[0] * 2), interpolation=cv2.INTER_AREA)
            u_t = cv2.resize(u_t, (u_t.shape[1] * 2, u_t.shape[0] * 2), interpolation=cv2.INTER_AREA)
            v_t = cv2.resize(v_t, (v_t.shape[1] * 2, v_t.shape[0] * 2), interpolation=cv2.INTER_AREA)
            u_t += u_d + U[i + 1]
            v_t += v_d + V[i + 1]

        ug = np.column_stack(np.where(ugol > 0.1 * ugol.max()))

        for p in ug:
            j, i = p
            print(p)
            mask = cv2.line(mask, (i, j), (int(i + (u_t[j, i])),  int(j + (v_t[j, i]))), (0, 255, 0), 2)
            frame11 = cv2.circle(frame11, (i, j), 1, (255, 0, 0), 1)
        img = cv2.add(frame11, mask)

        return img
