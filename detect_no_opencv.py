import numpy as np
import gaussian_filter_no_opencv as gauss
import skimage
import clusters


class VideoProcessingWithoutOpencv:

    def __init__(self):
        self.history_points = []

    def detect_without_opencv(self, frame1, frame2):
        img_first = frame1.astype(np.float32)
        img_second = frame2.astype(np.float32)

        difference = np.abs(img_first - img_second)
        gray_img = skimage.color.rgb2gray(difference)

        blur_img = gauss.GaussianFilter(9).gauss_blur(gray_img)
        threshold = 40
        threshold_img = 255 * (blur_img > threshold)

        contours = 255 * (np.abs(threshold_img[:-1, :] - threshold_img[1:, :]) > 0)

        clusters_x, clusters_y = clusters.find_clusters(contours)

        # Отрисовка рамок вокруг объектов
        # крайние точки рамок
        x_min, x_max, y_min, y_max = 0, 0, 0, 0

        trecker_y, trecker_x = np.zeros(len(clusters_x)), np.zeros(len(clusters_y))

        if len(clusters_y) != 0 and len(clusters_x) != 0:
            for i in range(len(clusters_x)):
                y_min = min(clusters_y[i])
                y_max = max(clusters_y[i])
                x_min = min(clusters_x[i])
                x_max = max(clusters_x[i])

                trecker_x[i] = int((x_min + x_max) // 2)
                trecker_y[i] = int((y_max + y_min) // 2)

                rr, cc = skimage.draw.rectangle_perimeter((x_min, y_min), end=(x_max, y_max), shape=frame1.shape)
                frame1[rr, cc] = (0, 255, 0)

            trecker_points = np.column_stack([trecker_y, trecker_x]).astype(np.float32)
            trecker_points = trecker_points.reshape(-1, 1, 2)

            self.history_points.append(trecker_points)

        return frame1, self.history_points
