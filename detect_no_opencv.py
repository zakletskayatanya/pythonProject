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
        threshold = 20
        threshold_img = 255 * (blur_img > threshold)

        contours = 255 * (np.abs(threshold_img[:-1, :] - threshold_img[1:, :]) > 0)
        cont = [contours[i] for i in range(0, len(contours), 20)]
        cont = np.array(cont)

        clust = clusters.predict(contours)
        clust = np.array(clust)

        # Отрисовка рамок вокруг объектов
        # крайние точки рамок
        x_min, x_max, y_min, y_max = 0, 0, 0, 0

        trecker_y, trecker_x = np.zeros(len(clust)), np.zeros(len(clust))

        if len(clust) != 0:
            for i in range(len(clust)):
                # x = [clust[i][j] for j in range(0, len(clust[i]), 2)]
                # y = [clust[i][j] for j in range(1, len(clust[i]), 2)]
                y_min = (clust[i][0][1])
                y_max = (clust[i][1][1])
                x_min = (clust[i][0][0])
                x_max = (clust[i][1][0])

                trecker_x[i] = int((x_min + x_max) // 2)
                trecker_y[i] = int((y_max + y_min) // 2)

                rr, cc = skimage.draw.rectangle_perimeter((x_min, y_min), end=(x_max, y_max), shape=frame1.shape)
                frame1[rr, cc] = (0, 255, 0)

            trecker_points = np.column_stack([trecker_y, trecker_x]).astype(np.float32)
            trecker_points = trecker_points.reshape(-1, 1, 2)

            self.history_points.append(trecker_points)

        return frame1, self.history_points, cont.astype('ubyte')
