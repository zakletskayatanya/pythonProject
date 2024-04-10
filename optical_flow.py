import math

import cv2
import numpy as np
import gaussian_filter_no_opencv as gauss
from scipy.signal import convolve2d


def func_norm(i, j):
    sigma = 3
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-1 / 2 * ((i - j) ** 2) / sigma ** 2)


class OpticalFlowProcessor:

    def __init__(self, frame1):
        self.history_points = []
        self.color = (0, 0, 255)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def calculate_optical_flow(self, frame1, frame2, treck, gradient_x1, gradient_y1, gradient_time, mask):

        w_size = 3


        # u_d = np.zeros(gradient_x.shape)
        # v_d = np.zeros(gradient_x.shape)
        U = []
        V = []
        # ugol = np.zeros(gradient_x1.shape)
        sobel_x = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
        sobel_y = np.array([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]])
        # func_gauss = list(map(func_norm, range(2*w_size), range(2*w_size)))
        # weight = np.array(list(func_gauss) * 2 * w_size)

        for iter in range(4, -1, -1):
            # print(iter)
            # gradient_x = cv2.resize(gradient_x, (840 // 2 ** iter, 480 // 2 ** iter), interpolation=cv2.INTER_AREA)
            # gradient_y = cv2.resize(gradient_y, (840 // 2 ** iter, 480 // 2 ** iter), interpolation=cv2.INTER_AREA)
            # gradient_time = cv2.resize(gradient_time, (840 // 2 ** iter, 480 // 2 ** iter),
            #                            interpolation=cv2.INTER_AREA)
            frame1_pyr = cv2.resize(frame1, (840 // 2 ** iter, 480 // 2 ** iter), interpolation=cv2.INTER_AREA)
            frame2_pyr = cv2.resize(frame2, (840 // 2 ** iter, 480 // 2 ** iter), interpolation=cv2.INTER_AREA)

            gray_img1 = cv2.cvtColor(frame1_pyr, cv2.COLOR_BGR2GRAY)
            gray_img2 = cv2.cvtColor(frame2_pyr, cv2.COLOR_BGR2GRAY)
            blur_img1 = gauss.GaussianFilter(7).gauss_blur(gray_img1)
            blur_img2 = gauss.GaussianFilter(7).gauss_blur(gray_img2)

            gradient_x = convolve2d(blur_img1, sobel_x, mode='same')
            gradient_y = convolve2d(blur_img1, sobel_y, mode='same')
            u = np.zeros(gradient_x1.shape)
            v = np.zeros(gradient_x1.shape)
            ugol = np.zeros(gradient_x1.shape)
            for rect in treck:
                y, x, h, w = rect
                y = y // 2 ** iter
                x = x // 2 ** iter
                h = h // 2 ** iter
                w = w // 2 ** iter
                for i in range(x + w_size, x + w - w_size):
                    for j in range(y + w_size, y + h - w_size):
                        Ix1 = gradient_x[i - w_size:i + w_size, j - w_size:j + w_size].flatten()
                        Iy1 = gradient_y[i - w_size:i + w_size, j - w_size:j + w_size].flatten()
                        It = blur_img2[i - w_size:i + w_size, j - w_size:j + w_size].flatten()
                        weight = func_norm(i, j)
                        M = weight * np.array([[np.sum((Ix1) ** 2), np.sum(Ix1 * Iy1)],
                                               [np.sum(Iy1 * Ix1), np.sum((Iy1) ** 2)]])
                        b = -weight * np.array([np.sum(Ix1 * It), np.sum(Iy1 * It)])
                        if np.linalg.det(M) != 0:
                            # continue
                            k = 0.04
                            R = np.linalg.det(M) - k * np.trace(M) ** 2
                            # print(R)
                            if R > k:
                                ugol[i, j] = R
                                uv = np.linalg.solve(M, b)
                                u[i, j] = uv[0]
                                v[i, j] = uv[1]
            U.append(u)
            V.append(v)
        for i in range(len(U)-1):
            u_t, v_t = U[i], V[i]
            u_d = u_t * 2
            v_d = v_t * 2
            u_t = u_d + U[i + 1]
            v_t = v_d + V[i + 1]

        # frame1 = cv2.resize(frame1, (840, 480), interpolation=cv2.INTER_AREA)
        # u_t = u + u_d
        # v_t = v + v_d
        ug = np.column_stack(np.where(ugol > 0.01 * ugol.max()))

        for p in ug:
            j, i = p
            # print(u[j,i], v[j,i])
            mask = cv2.line(mask, (i, j), (i + int(np.ceil(u_t[j, i])), j + int(np.ceil(v_t[j, i]))), (0, 255, 0), 2)
            frame1 = cv2.circle(frame1, (i, j), 1, (255, 0, 0), 1)
        img = cv2.add(frame1, mask)

        return img
