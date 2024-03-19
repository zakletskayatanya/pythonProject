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
        dim = (960, 540)
        frame1 = cv2.resize(frame1, dim, interpolation=cv2.INTER_AREA)
        frame2 = cv2.resize(frame2, dim, interpolation=cv2.INTER_AREA)
        diff = cv2.absdiff(frame1, frame2)
        gray_img1 = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        blur_img1 = gauss.GaussianFilter(5).gauss_blur(gray_img1)



        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

        gradient_x = convolve2d(blur_img1, sobel_x, mode='same')
        gradient_y = convolve2d(blur_img1, sobel_y, mode='same')

        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_direction = np.arctan2(gradient_y, gradient_x) * 180 / math.pi

        for i in range(1, gradient_direction.shape[0] - 1):
            for j in range(1, gradient_direction.shape[1] - 1):
                grad = gradient_direction[i, j]
                local_max = gradient_magnitude[i, j]
                if local_max == 0:
                    continue

                neighborhood_x = [gradient_magnitude[i + k, j] for k in range(-1, 2)]
                neighborhood_y = [gradient_magnitude[i , j+k] for k in range(-1, 2)]
                neighborhood_diag1 = [gradient_magnitude[i + k, j + k] for k in range(-1, 2)]
                neighborhood_diag2 = [gradient_magnitude[i + k, j - k] for k in range(-1, 2)]

                if -20 < grad < 20 or 160 < grad < 200 or -160 < grad < -200:
                    gradient_magnitude[i, j] = np.where(local_max == max(neighborhood_x), local_max, 0)
                elif 70 < grad < 110 or -70 < grad < -110 or -250 < grad < -290 or 250 < grad < 290:
                    gradient_magnitude[i, j] = np.where(local_max == max(neighborhood_y), local_max, 0)
                elif 25 < grad < 65 or 205 < grad < 245 or -25 < grad < -65 or -205 < grad < -245:
                    gradient_magnitude[i, j] = np.where(local_max == max(neighborhood_diag1), local_max, 0)
                elif 115 < grad < 155 or 295 < grad < 335 or -115 < grad < -155 or -295 < grad < -335:
                    gradient_magnitude[i, j] = np.where(local_max == max(neighborhood_diag2), local_max, 0)

        top_threshhold = 240
        low_threshhold = 60

        gradient_magnitude = np.where(gradient_magnitude >= top_threshhold, 255, gradient_magnitude)
        gradient_magnitude = np.where(gradient_magnitude <= low_threshhold, 0, gradient_magnitude)
        gradient_magnitude = np.where((low_threshhold <= gradient_magnitude) & (gradient_magnitude <= top_threshhold),
                                      100, gradient_magnitude)

        # trassirovka = np.column_stack(np.where(gradient_magnitude == 100))
        #
        # for el in trassirovka:
        #     i, j = el
        #     neighborhood = [gradient_magnitude[i+k, j+k] for k in range(-1,1)]
        #     gradient_magnitude[i, j] = np.where(neighborhood == 255, 255, 0)

        # cc = clusters_dbscan.dbscan_naive(gradient_magnitude, 20, 3)
        # clust = clusters.find_objects(gradient_magnitude)
        # print(cc)
        cc = cluusters_ierarh.find_clusters(gradient_magnitude)

        if cc is not None:
            # trecker_y, trecker_x = np.zeros(len(clust)), np.zeros(len(clust))
            # x_min, x_max, y_min, y_max = 0, 0, 0, 0
            for cluster in cc:
                xx, yy, w, h = cv2.boundingRect(np.array(cluster))

                if cv2.contourArea(np.array(cluster)) > 20:  # условие при котором площадь выделенного объекта меньше 700 px
                    cv2.rectangle(frame1, (xx, yy), (xx + w, yy+h), (0, 255, 0), 2)
            # trecker_points = np.column_stack([trecker_y, trecker_x]).astype(np.float32)
            # trecker_points = trecker_points.reshape(-1, 1, 2)

            # self.history_points.append(trecker_points)

        return frame1, self.history_points, gradient_magnitude.astype('ubyte')
