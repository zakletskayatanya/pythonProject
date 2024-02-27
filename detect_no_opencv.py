import numpy as np
import gaussian_filter_no_opencv as gauss
import skimage
import clusters
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

        # difference = np.abs(img_first - img_second)
        # gray_img = skimage.color.rgb2gray(difference)

        diff = cv2.absdiff(frame1,
                           frame2)  # нахождение разницы двух кадров, которая проявляется лишь при изменении одного из них, т.е. с этого момента наша программа реагирует на любое движение.
        gray_img = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        blur_img = gauss.GaussianFilter(9).gauss_blur(gray_img)
        threshold = 20
        threshold_img = np.where(blur_img > threshold, 255, 0)


        # contours = 255 * (np.abs(threshold_img[:-1, :] - threshold_img[1:, :]) > 0)
        # cont = [threshold_img[i] for i in range(0, len(threshold_img), 30)]
        # cont = np.array(cont)
        # clust = True

        # clust = clusters.predict(contours)
        # clust = np.array(clust)

        clust = clusters_dbscan.dbscan_naive(threshold_img, 20, 7)

        # Отрисовка рамок вокруг объектов
        # крайние точки рамок

        if clust is not None:
            clust.pop(-10)
            # print(clust)
            # trecker_y, trecker_x = np.zeros(len(clust)), np.zeros(len(clust))
            # x_min, x_max, y_min, y_max = 0, 0, 0, 0
            for key, cluster in clust.items():
                cluster_array = np.array(cluster)
                point_min = np.min(cluster_array, axis=0)
                point_max = np.max(cluster_array, axis=0)
                # x = cluster_array[:, 0]
                # y = cluster_array[:, 1]
                # x = [clust[i][j][0] for j in range(len(clust[i]))]
                # y = [clust[i][j][1] for j in range(len(clust[i]))]
                y_min = point_min[1]
                y_max = point_max[1]
                x_min = point_min[0]
                x_max = point_max[0]

                # trecker_x[i] = int((x_min + x_max) // 2)
                # trecker_y[i] = int((y_max + y_min) // 2)

                rr, cc = skimage.draw.rectangle_perimeter((x_min, y_min), end=(x_max, y_max), shape=frame1.shape)
                frame1[rr, cc] = (0, 255, 0)

            # trecker_points = np.column_stack([trecker_y, trecker_x]).astype(np.float32)
            # trecker_points = trecker_points.reshape(-1, 1, 2)

            # self.history_points.append(trecker_points)

        return frame1, self.history_points, threshold_img.astype('ubyte')
