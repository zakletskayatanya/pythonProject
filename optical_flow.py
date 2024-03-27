import cv2
import numpy as np
import gaussian_filter_no_opencv as gauss
from scipy.signal import convolve2d


class OpticalFlowProcessor:

    def __init__(self, frame1):
        self.history_points = []
        self.color = (0, 0, 255)
        self.mask = np.zeros_like(frame1)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def calculate_optical_flow(self, frame1, frame2, treck, gradient_x, gradient_y, gradient_time, points):

        p0 = points
        mask = np.zeros_like(frame2)
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        sobel_y = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])

        gray_img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray_img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        blur_img1 = gauss.GaussianFilter(5).gauss_blur(gray_img1)
        blur_img2 = gauss.GaussianFilter(5).gauss_blur(gray_img2)
        gray_img = cv2.absdiff(blur_img1, blur_img2)
        # gray_img = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # gray_img = gray_img.astype(np.float32)

        gradient_t = gray_img

        gradient_x_1 = convolve2d(blur_img1, sobel_x, mode='same')
        gradient_y_1 = convolve2d(blur_img1, sobel_y, mode='same')

        gradient_x_2 = convolve2d(blur_img2, sobel_x, mode='same')
        gradient_y_2 = convolve2d(blur_img2, sobel_y, mode='same')

        w_size = 5

        # M = np.array([[np.sum(gradient_x_1 ** 2), np.sum(gradient_x_1 * gradient_y)],
        #                                            [np.sum(gradient_x_1 * gradient_y_1), np.sum(gradient_y_1 ** 2)]])
        # b = -np.array([np.sum(gradient_x_2), np.sum(gradient_y_2)])
        # uv = -np.matmul(np.linalg.inv(M), b)

        u = np.zeros(gray_img1.shape)
        v = np.zeros(gray_img1.shape)
        ugol = np.zeros(gray_img1.shape)
        for rect in treck:
            y, x, h, w = rect
            for i in range(x+w_size, x+w - w_size):
                for j in range(y+w_size, y+h - w_size):
                    Ix1 = gradient_x_1[i - w_size:i + w_size + 1, j - w_size:j + w_size + 1].flatten()
                    Iy1 = gradient_y_1[i - w_size:i + w_size + 1, j - w_size:j + w_size + 1].flatten()
                    It = gradient_t[i - w_size:i + w_size + 1, j - w_size:j + w_size + 1].flatten()
                    # Ix2 = gradient_x_2[i - w:i + w + 1, j - w:j + w + 1].flatten()
                    # Iy2 = gradient_y_2[i - w:i + w + 1, j - w:j + w + 1].flatten()
                    M = np.array([[np.sum(Ix1 ** 2), np.sum(Ix1 * Iy1)],
                                  [np.sum(Iy1 * Ix1), np.sum(Iy1 ** 2)]])
                    b = -np.array([np.sum(Ix1*It), np.sum(Iy1*It)])
                    if np.linalg.det(M) <= 0:
                        continue
                    uv = -np.matmul(np.linalg.inv(M), b)
                    u[i, j] = uv[0]
                    v[i, j] = uv[1]
                    # k = 0.15
                    # R = np.linalg.det(M)-k*np.trace(M)**2
                    # if R > k:
                    # ugol[i, j] = R
        p1 = np.sqrt(u**2+v**2)
        # ug = np.column_stack(np.where(ugol > 0.1*np.max(ugol)))

        for p in p0:
            i, j = p
            print(p)
            print()
            print(round(i+u[i, j]), round(j+v[i, j]))
            mask = cv2.line(mask, (i, j), (round(i+u[i, j]), round(j+v[i, j])), (0, 0, 255), 2)
            frame1 = cv2.circle(frame1, (i, j), 1, (0, 0, 255), -1)
        img = cv2.add(frame1, mask)
        points=p1
        # E = (f_x*u+f_y*v)**2
        # print(np.max(E), np.min(E))

        return img, gradient_t
