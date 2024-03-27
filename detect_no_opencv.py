import math

import numpy as np
import gaussian_filter_no_opencv as gauss
import skimage
import clusters
from scipy.signal import convolve2d
import clusters_dbscan
import cv2
import cluusters_ierarh
from scipy.ndimage.filters import maximum_filter
from sklearn.cluster import DBSCAN


class VideoProcessingWithoutOpencv:

    def __init__(self):
        self.history_points = []

    def detect_without_opencv(self, frame1, frame2):
        diff = cv2.absdiff(frame1, frame2)
        gray_img1 = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # gray_img1 = gray_img1.astype(np.float32)

        blur_img1 = gauss.GaussianFilter(5).gauss_blur(gray_img1)

        sobel_x = np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])

        sobel_y = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])

        sobel_time = np.array([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]])

        gradient_x = convolve2d(blur_img1, sobel_x, mode='same')
        gradient_y = convolve2d(blur_img1, sobel_y, mode='same')
        gradient_time = convolve2d(blur_img1, sobel_time, mode='same')

        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_direction = np.arctan2(gradient_y, gradient_x) * 180 / math.pi
        gradient_direction = np.round(gradient_direction / 45) * 45  # округление до красности 45

        suppressed = np.zeros_like(gradient_magnitude)
        angle = gradient_direction

        neighbors = np.zeros_like(gradient_magnitude)

        # Соседи по горизонтали
        neighbors[:, :-1] = gradient_magnitude[:, 1:]
        neighbors[:, 1:] = gradient_magnitude[:, :-1]

        # Соседи по вертикали
        neighbors[:-1, :] = np.maximum(neighbors[:-1, :], gradient_magnitude[1:, :])
        neighbors[1:, :] = np.maximum(neighbors[1:, :], gradient_magnitude[:-1, :])

        # Соседи по диагонали (все четыре направления)
        neighbors_diag1 = np.zeros_like(gradient_magnitude)
        neighbors_diag2 = np.zeros_like(gradient_magnitude)

        neighbors_diag1[:-1, :-1] = gradient_magnitude[1:, 1:]  # Сосед по диагонали влево вниз
        neighbors_diag1[1:, 1:] = np.maximum(neighbors_diag1[1:, 1:], gradient_magnitude[:-1, :-1])

        neighbors_diag2[:-1, 1:] = gradient_magnitude[1:, :-1]  # Сосед по диагонали вправо вниз
        neighbors_diag2[1:, :-1] = np.maximum(neighbors_diag2[1:, :-1], gradient_magnitude[:-1, 1:])

        mask_horizontal = np.logical_or(angle == 0, angle == 180, angle == -180)  # Горизонтальное направление
        mask_diag1 = np.logical_or(angle == 45, angle == -135)  # Диагональное направление 1
        mask_vertical = np.logical_or(angle == 90, angle == -90)  # Вертикальное направление
        mask_diag2 = np.logical_or(angle == 135, angle == -45)  # Диагональное направление 2

        max_neighbors = np.zeros_like(gradient_magnitude)
        max_neighbors[mask_horizontal] = neighbors[mask_horizontal]
        max_neighbors[mask_diag1] = neighbors[mask_diag1]
        max_neighbors[mask_vertical] = neighbors[mask_vertical]
        max_neighbors[mask_diag2] = neighbors[mask_diag2]

        suppressed = np.where(gradient_magnitude >= max_neighbors, gradient_magnitude, suppressed)

        top_threshhold = 120
        low_threshhold = 80

        suppressed = np.where(suppressed >= top_threshhold, 255, suppressed)
        suppressed = np.where(suppressed <= low_threshhold, 0, suppressed)
        suppressed = np.where((low_threshhold <= suppressed) & (suppressed <= top_threshhold),
                              100, suppressed)

        # trassirovka = np.column_stack(np.where(gradient_magnitude == 100))
        #
        # for el in trassirovka:
        #     i, j = el
        #     neighborhood = [suppressed[i+k, j+k] for k in range(-1,1)]
        #     suppressed[i, j] = np.where(neighborhood == 255, 255, 0)

        # cc = clusters_dbscan.dbscan_naive(gradient_magnitude, 20, 3)
        # clust = clusters.find_objects(gradient_magnitude)
        # print(cc)
        cc = cluusters_ierarh.find_clusters(suppressed)
        trecker_rect = []
        if cc is not None:

            for cluster in cc:
                xx, yy, w, h = cv2.boundingRect(np.array(cluster))
                trecker_rect.append([xx, yy, w, h])
                self.history_points.append([(xx+w)//2, (yy+h)//2])
                # if cv2.contourArea(
                #         np.array(cluster)) > 50:  # условие при котором площадь выделенного объекта меньше 700 px
                    # cv2.rectangle(frame1, (xx, yy), (xx + w, yy + h), (0, 255, 0), 2)

        return frame1, trecker_rect, gradient_x, gradient_y, gradient_time, suppressed.astype('ubyte'), self.history_points
