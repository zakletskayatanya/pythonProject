import cv2
import numpy as np
import math
from scipy.signal import convolve2d


class OpticalFlowProcessor:

    def __init__(self, frame1):
        self.history_points = []
        self.color = (0, 0, 255)
        self.mask = np.zeros_like(frame1)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def calculate_optical_flow(self, frame1, frame2, treck, gragient_x, gradient_y, gradient_time):

        kernel_x = np.array([[-1, 1],[-1, 1]])
        kernel_y = np.array([[-1, -1], [1, 1]])
        kernel_t = np.array([[1, 1], [-1, 1]])

        gray_img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray_img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        w_size = 2

        f_x = convolve2d(gray_img1, kernel_x, mode='same')
        f_y = convolve2d(gray_img1, kernel_y, mode='same')
        f_t = convolve2d(gray_img2, kernel_t, mode='same') + convolve2d(gray_img1, -kernel_t, mode='same')

        u = np.zeros(gray_img1.shape)
        v = np.zeros(gray_img1.shape)
        for k in treck:
            y, x, h, w = k
            for i in range(x+w_size, x+w - w_size):
                for j in range(y+w_size, y+h - w_size):
                    Ix = f_x[i-w_size:i+w_size+1, j-w_size:j+w_size+1].flatten()
                    Iy = f_y[i-w_size:i+w_size+1, j-w_size:j+w_size+1].flatten()
                    It = f_t[i-w_size:i+w_size+1, j-w_size:j+w_size+1].flatten()
                    b = -It
                    print(b)
                    A = np.column_stack((Ix, Iy))
                    if np.linalg.det(np.matmul(np.transpose(A), A)) == 0:
                        continue
                    nu = np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)), np.matmul(np.transpose(A), b))
                    u[i, j] = nu[0]
                    v[i, j] = nu[1]

        # print(np.nonzero(u))
        return (u, v)
