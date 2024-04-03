import math

import cv2
import numpy as np
import gaussian_filter_no_opencv as gauss
from scipy.signal import convolve2d


def func_norm(i, j):
    return 1 / 2 / np.pi / 4 * np.exp(-(i ** 2 + j ** 2 / 2 / 4))


class OpticalFlowProcessor:

    def __init__(self, frame1):
        self.history_points = []
        self.color = (0, 0, 255)
        self.mask = np.zeros_like(frame1)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def calculate_optical_flow(self, frame1, frame2, treck, gradient_x, gradient_y, gradient_time, con):

        mask = np.zeros_like(frame2)
        sobel_y = np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])

        sobel_x = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])

        gray_img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray_img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        blur_img1 = gauss.GaussianFilter(5).gauss_blur(gray_img1)
        blur_img2 = gauss.GaussianFilter(5).gauss_blur(gray_img2)

        gray_img = cv2.absdiff(gray_img1, gray_img2)
        # gray_img = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # gray_img = gray_img.astype(np.float32)

        gradient_t = con

        gradient_x_1 = convolve2d(gray_img1, sobel_x, mode='same')
        gradient_y_1 = convolve2d(gray_img1, sobel_y, mode='same')

        # blur_gr1 = gauss.GaussianFilter(5).gauss_blur(gradient_x_1)
        # blur_gr2 = gauss.GaussianFilter(5).gauss_blur(gradient_y_1)
        #
        # gradient_x_2 = convolve2d(blur_img1, sobel_x, mode='same')
        # gradient_y_2 = convolve2d(blur_img1, sobel_y, mode='same')

        w_size = 15

        u = np.zeros(gray_img1.shape)
        v = np.zeros(gray_img1.shape)
        ugol = np.zeros(gray_img1.shape)
        func_gauss = list(map(func_norm, range(2*w_size), range(2*w_size)))
        weight = np.array(list(func_gauss) * 2 * w_size)
        for rect in treck:
            y, x, h, w = rect
            for i in range(x+w_size, x+w - w_size):
                for j in range(y+w_size, y+h - w_size):
                    Ix1 = gradient_x_1[i - w_size:i + w_size, j - w_size:j + w_size].flatten()
                    Iy1 = gradient_y_1[i - w_size:i + w_size, j - w_size:j + w_size].flatten()
                    It = gray_img[i - w_size:i + w_size, j - w_size:j + w_size].flatten()
                    # for ii in range(i-w_size, i + w_size + 1):
                    #     count_j = 0
                    #     for jj in range(j-w_size, j + w_size + 1):
                    #         M00 += -(Ix1[count_i, count_j]*1/2/np.pi/4*np.exp(-(count_i**2+count_j**2/2/4))) ** 2
                    #         M11 += -(Iy1[count_i, count_j]*1/2/np.pi/4*np.exp(-(count_i**2+count_j**2/2/4))) ** 2
                    #         M01 += (Ix1[count_i, count_j]*Iy1[count_i, count_j]*1/2/np.pi/4*np.exp(-(count_i**2+count_j**2/2/4)))
                    #         b0 = Ix1[count_i, count_j] * It[count_i, count_j]
                    #         b1 = Iy1[count_i, count_j] * It[count_i, count_j]
                    #         count_j += 1
                    #     count_i += 1

                    # M = np.array([[M00, M01],
                    #               [M01, M11]])
                    # b = -np.array([b0, b1])

                    M = np.array([[np.sum((weight*Ix1) ** 2), np.sum(weight*Ix1 * Iy1)],
                                  [np.sum(weight*Iy1 * Ix1), np.sum((weight*Iy1) ** 2)]])
                    b = -np.array([np.sum(Ix1 * It), np.sum(Iy1 * It)])
                    if np.linalg.det(M) <= 0:
                        continue
                    uv = np.matmul(np.linalg.inv(M), b)
                    u[i, j] = uv[0]
                    v[i, j] = uv[1]
                    k = 0.15
                    R = np.linalg.det(M)-k*np.trace(M)**2
                    if R > k:
                        ugol[i, j] = R
        # p1 = np.sqrt(u**2+v**2)
        print(u[156,214], v[156,214])

        ug = np.column_stack(np.where(ugol > 0.01*ugol.max()))

        for p in ug:
            j, i = p
            # print(p)
            # print(u[i,j], v[i,j])
            mask = cv2.line(mask, (i, j), (math.ceil(i+u[i, j]), math.ceil(j+v[i, j])), (0, 0, 255), 2)
            frame1 = cv2.circle(frame1, (i, j), 1, (0, 0, 255), -1)
        img = cv2.add(frame1, mask)
        # points=p1
        # E = (f_x*u+f_y*v)**2
        # print(np.max(E), np.min(E))

        return img, gradient_t
