import numpy as np
import gaussian_filter_no_opencv as gauss
import skimage
import clusters
import clusters_dbscan
from sklearn.cluster import DBSCAN


class VideoProcessingWithoutOpencv:

    def __init__(self):
        self.history_points = []

    def detect_without_opencv(self, frame1, frame2):
        img_first = frame1.astype(np.float32)
        img_second = frame2.astype(np.float32)

        difference = np.abs(img_first - img_second)
        gray_img = skimage.color.rgb2gray(difference)

        blur_img = gauss.GaussianFilter(9).gauss_blur(gray_img)
        threshold = 30
        threshold_img = 255 * (blur_img > threshold)

        contours = 255 * (np.abs(threshold_img[:-1, :] - threshold_img[1:, :]) > 0)
        cont = [contours[i] for i in range(0, len(contours), 50)]
        cont = np.array(cont)
        clust = True

        # clust = clusters.predict(contours)
        # clust = np.array(clust)

        # clust = clusters_dbscan.dbscan_naive(contours, 50, 3)
        x, y = np.nonzero(contours == 255)  # координаты контуров
        if len(x) < 1 or len(y) < 1 or len(x) > 10000 or len(y) > 10000:
            clust = None

        if clust is not None:

            points = np.column_stack((x, y))

            P = [points[i] for i in range(len(points))]
            clust = DBSCAN(eps=40, min_samples=5).fit(P)
            l = clust.labels_
            ll = set(l)
            print(len(ll))

            trecker_y, trecker_x = np.zeros(len(ll)-1), np.zeros(len(ll)-1)

            for i in range(len(ll) - 1):
                index = np.array(np.where(l == i))
                point = []
                for j in range(len(index[0])):
                    point.append(P[index[0, j]])
                x = [point[k][0] for k in range(len(point))]
                y = [point[k][1] for k in range(len(point))]

                y_min = min(y)
                y_max = max(y)
                x_min = min(x)
                x_max = max(x)

                trecker_x[i] = int((x_min + x_max) // 2)
                trecker_y[i] = int((y_max + y_min) // 2)

                rr, cc = skimage.draw.rectangle_perimeter((x_min, y_min), end=(x_max, y_max), shape=frame1.shape)
                frame1[rr, cc] = (0, 255, 0)
            trecker_points = np.column_stack([trecker_y, trecker_x]).astype(np.float32)
            trecker_points = trecker_points.reshape(-1, 1, 2)

            self.history_points.append(trecker_points)


        # print(clust)

        # Отрисовка рамок вокруг объектов
        # крайние точки рамок

        # if clust is not None:
        #     print(len(clust))
        #     trecker_y, trecker_x = np.zeros(len(clust)), np.zeros(len(clust))
        #     # x_min, x_max, y_min, y_max = 0, 0, 0, 0
        #     for i in range(len(clust)):
        #         x = [clust[i][j][0] for j in range(len(clust[i]))]
        #         y = [clust[i][j][1] for j in range(len(clust[i]))]
        #         y_min = min(y)
        #         y_max = max(y)
        #         x_min = min(x)
        #         x_max = max(x)
        #
        #         trecker_x[i] = int((x_min + x_max) // 2)
        #         trecker_y[i] = int((y_max + y_min) // 2)
        #
        #         rr, cc = skimage.draw.rectangle_perimeter((x_min, y_min), end=(x_max, y_max), shape=frame1.shape)
        #         frame1[rr, cc] = (0, 255, 0)
        #
        #     trecker_points = np.column_stack([trecker_y, trecker_x]).astype(np.float32)
        #     trecker_points = trecker_points.reshape(-1, 1, 2)
        #
        #     self.history_points.append(trecker_points)

        return frame1, self.history_points, cont.astype('ubyte')
