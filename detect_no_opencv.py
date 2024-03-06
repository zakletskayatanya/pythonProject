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

        # difference = np.abs(img_first - img_second)
        # gray_img = skimage.color.rgb2gray(difference)

          # нахождение разницы двух кадров, которая проявляется лишь при изменении одного из них, т.е. с этого момента наша программа реагирует на любое движение.
        # gray_img = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        blur_img = gauss.GaussianFilter(5).gauss_blur(diff)

        sobel_x = np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])

        sobel_y = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])

        gradient_x = convolve2d(blur_img, sobel_x, mode='same')
        gradient_y = convolve2d(blur_img, sobel_y, mode='same')
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        # gradient_direction = np.arctan2(gradient_y, gradient_x)

        gradient_magnitude = np.where(gradient_magnitude > 120, 255, 0)


        # for i in range(1, blur_img.shape[0]-1):
        #     for j in range(1, blur_img.shape[1]-1):
        #         Gx[i, j] = np.sum(np.multiply(blur_img[(i - 1):(i + 2), (j - 1):(j + 2)], sobel_x))
        #         Gy[i, j] = np.sum(np.multiply(blur_img[(i - 1):(i + 2), (j - 1):(j + 2)], sobel_y))


        # threshold = 20
        # threshold_img = np.where(blur_img > threshold, 255, 0)
        _, threshold_img = cv2.threshold(blur_img, 25, 255, cv2.THRESH_BINARY)

        # contours = 255 * (np.abs(threshold_img[:-1, :] - threshold_img[1:, :]) > 0)
        # cont = [threshold_img[i] for i in range(0, len(threshold_img), 30)]
        # cont = np.array(cont)
        # clust = True

        # clust = clusters.predict(contours)
        # clust = np.array(clust)

        cc = clusters_dbscan.dbscan_naive(gradient_magnitude, 20, 3)
        # clust = clusters.find_objects(gradient_magnitude)
        # print(cc)
        clust = cluusters_ierarh.find_clusters(gradient_magnitude)
        print(clust)

        if cc is not None:
            # clust.pop(-10)
            # print(clust)
            # trecker_y, trecker_x = np.zeros(len(clust)), np.zeros(len(clust))
            # x_min, x_max, y_min, y_max = 0, 0, 0, 0
            for key, cluster in cc.items():
                cluster_array = np.array(cluster)
                yy, xx, h, w = cv2.boundingRect(cluster_array)

                # if len(cluster_array) < 10:
                #     continue
                if cv2.contourArea(cluster_array) < 50:  # условие при котором площадь выделенного объекта меньше 700 px
                    continue
                point_min = np.min(cluster_array, axis=0)
                point_max = np.max(cluster_array, axis=0)
                cv2.rectangle(frame1, (xx, yy), (xx + w, yy+h), (0, 255, 0), 2)

                # cv2.rectangle(frame1, (point_min[1], point_min[0]), (point_max[1], point_max[0]), (0, 255, 0), 2)

            # trecker_points = np.column_stack([trecker_y, trecker_x]).astype(np.float32)
            # trecker_points = trecker_points.reshape(-1, 1, 2)

            # self.history_points.append(trecker_points)

        return frame1, self.history_points, gradient_magnitude.astype('ubyte')
