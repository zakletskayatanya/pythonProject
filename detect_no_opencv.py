import math

import numpy as np
import gaussian_filter_no_opencv as gauss
import skimage
import clusters
from scipy.signal import convolve2d
import clusters_dbscan
import cv2
import cluusters_ierarh
from sklearn.cluster import DBSCAN


class VideoProcessingWithoutOpencv:

    def __init__(self):
        self.history_points = []

    def detect_without_opencv(self, frame1, frame2):
        # img_first = frame1.astype(np.float32)
        # img_second = frame2.astype(np.float32)
        dim = (960, 540)
        frame1 = cv2.resize(frame1, dim, interpolation=cv2.INTER_AREA)
        frame2 = cv2.resize(frame2, dim, interpolation=cv2.INTER_AREA)
        gray_img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray_img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_img1, gray_img2)

        blur_img = gauss.GaussianFilter(5).gauss_blur(diff)

        sobel_x = np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])

        sobel_y = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])

        gradient_x = convolve2d(blur_img, sobel_x, mode='same')
        gradient_y = convolve2d(blur_img, sobel_y, mode='same')

        # Gx = np.zeros_like(blur_img)
        # Gy = np.zeros_like(blur_img)
        #
        # for i in range(1, blur_img.shape[0]-1):
        #     for j in range(1, blur_img.shape[1]-1):
        #         Gx[i, j] = np.sum(np.multiply(blur_img[(i - 1):(i + 2), (j - 1):(j + 2)], sobel_x))
        #         Gy[i, j] = np.sum(np.multiply(blur_img[(i - 1):(i + 2), (j - 1):(j + 2)], sobel_y))
        # gradient_magnitude = np.sqrt(Gx ** 2 + Gy ** 2)

        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

        gradient_direction = np.arctan2(gradient_y, gradient_x) * 180 / math.pi
        print(gradient_direction.shape)

        for (i,j), grad in np.ndenumerate(gradient_direction):
            if i == 0 or j ==0 or i == gradient_direction.shape[0]-1 or j == gradient_direction.shape[1]-1:
                continue
            if -10 < grad < 10 or 170 < grad < 190:
                gradient_direction[i,j] = np.where(grad == np.max([gradient_direction[i,j], gradient_direction[i-1,j],gradient_direction[i+1,j]]), grad, 0)
                continue
            if 35 < grad < 55 or 215 < grad < 235:
                gradient_direction[i,j] = np.where(grad == np.max([gradient_direction[i,j], gradient_direction[i-1,j+1],gradient_direction[i+1,j-1]]), grad, 0)
                continue
            if 125 < grad < 145 or 305 < grad < 325:
                gradient_direction[i,j] = np.where(grad == np.max([gradient_direction[i,j], gradient_direction[i+1,j+1],gradient_direction[i-1,j-1]]), grad, 0)
                continue

        top_threshhold = 200
        low_threshhold = 30

        gradient_magnitude = np.where(gradient_magnitude >= top_threshhold, 255, gradient_magnitude)
        gradient_magnitude = np.where(gradient_magnitude <= low_threshhold, 0, gradient_magnitude)
        # проверить это дома
        gradient_magnitude = np.where(low_threshhold >= gradient_magnitude and gradient_magnitude <= top_threshhold, 100, gradient_magnitude)

        # cc = clusters_dbscan.dbscan_naive(gradient_magnitude, 20, 3)
        # clust = clusters.find_objects(gradient_magnitude)
        # print(cc)
        cc = cluusters_ierarh.find_clusters(gradient_magnitude)

        if cc is not None:
            # trecker_y, trecker_x = np.zeros(len(clust)), np.zeros(len(clust))
            # x_min, x_max, y_min, y_max = 0, 0, 0, 0
            for cluster in cc:
                # cluster_array = np.array(cluster)
                xx, yy, w, h = cv2.boundingRect(np.array(cluster))

                if cv2.contourArea(np.array(cluster)) > 25:  # условие при котором площадь выделенного объекта меньше 700 px
                    cv2.rectangle(frame1, (xx, yy), (xx + w, yy+h), (0, 255, 0), 2)
            # trecker_points = np.column_stack([trecker_y, trecker_x]).astype(np.float32)
            # trecker_points = trecker_points.reshape(-1, 1, 2)

            # self.history_points.append(trecker_points)

        return frame1, self.history_points, gradient_magnitude.astype('ubyte')
