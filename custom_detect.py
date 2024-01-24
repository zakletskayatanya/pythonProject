import random
import math
import cv2  # импорт модуля cv2
import skimage
import numpy as np
import Gaussian_filter_no_opencv as gauss
from scipy.ndimage import gaussian_filter
import sklearn
from sklearn import cluster
import itertools

# cap = cv2.VideoCapture("http://192.168.217.103/mjpg/video.mjpg")  # видео поток с веб камеры
cap = cv2.VideoCapture("IMG_8546.MP4")  # видео поток с веб камеры

# cap.set(3, 200)  # установка размера окна
# cap.set(4, 300)

ret, frame1 = cap.read()
ret1, frame2 = cap.read()

while cap.isOpened():  # метод isOpened() выводит статус видеопотока

    # https://webtort.ru/%D0%BA%D0%B0%D0%BA-%D0%BD%D0%B0%D0%B9%D1%82%D0%B8-%D0%BE%D1%82%D0%BB%D0%B8%D1%87%D0%B8%D1%8F-%D0%BD%D0%B0-%D0%BA%D0%B0%D1%80%D1%82%D0%B8%D0%BD%D0%BA%D0%B0%D1%85-%D1%81-%D0%BF%D0%BE%D0%BC%D0%BE%D1%89/?ysclid=lpmzcgk0un225441379
    img_f1 = frame1.astype(np.float32)  # разница 2х кадров
    img_f2 = frame2.astype(np.float32)
    diff_custom = np.abs(img_f1 - img_f2)

    # diff_custom = diff_custom.astype(np.ubyte)
    gray_custom = skimage.color.rgb2gray(diff_custom)  # черно белый кадр

    blur_custom = gauss.GaussianFilter(7).gauss_blur(gray_custom)
    # blur_custom = blur_custom.astype(np.ubyte)
    # thresh_custom = np.clip(blur_custom, 10, 255)
    threshold = 20  # переход к бинарному изображению
    thresh_custom = 255 * (blur_custom > threshold)
    thresh_custom = thresh_custom.astype(np.ubyte)

    # contours = skimage.measure.find_contours(thresh_custom)
    contours1 = thresh_custom[:-1, :]
    contours2 = thresh_custom[1:, :]
    contours = 255 * (np.abs(contours1 - contours2) > 0)

    x, y = np.where(contours == 255)
    xx = np.where(contours == 255)
    # clust = cluster.AgglomerativeClustering(affinity="euclidean").fit(X)

    perem_a = []
    perem_b = []
    if len(x) > 0 and len(y) > 0:

        a, b = x, y
        # R = np.zeros((len(a) - 1, len(b) - 1))
        # R += 1000000000000

        rmin = -100
        klasters_a = [[el] for el in a.tolist()]
        klasters_b = [[el] for el in b.tolist()]
        # print(klasters_b)
        count = 1
        while rmin < 70:

            # if count == 0:
            #     count += 1
            #     for i in range(R.shape[0] - 1):
            #         for j in range(i + 1, R.shape[1]):
            #             R[i, j] = math.sqrt((a[i] - a[j]) ** 2 + (b[i] - b[j]) ** 2)
            #
            #     rmin = R.min()
            #     for i in range(R.shape[0] - 1):
            #         perem_a.append(a[i])
            #         perem_b.append(b[i])
            #         for j in range(i + 1, R.shape[1]):
            #             if rmin == R[i, j]:
            #                 perem_a.append(a[j])
            #                 perem_b.append(b[j])
            #                 R[i, j] = 1000000000000
            #         klasters_a.append(perem_a)
            #         klasters_b.append(perem_b)
            #         perem_a = []
            #         perem_b = []

            R = np.zeros((len(klasters_a), len(klasters_b)))
            R += 10000

            # if count > 0:
                # print(klasters_b)
            for i in range(len(klasters_a)):
                for j in range(i + 1, len(klasters_b)):
                    R[i, j] = math.sqrt(
                            (sum(klasters_a[i]) / len(klasters_a[i]) - sum(klasters_a[j]) / len(klasters_a[j])) ** 2 + (
                                    sum(klasters_b[i]) / len(klasters_b[i]) - sum(klasters_b[j]) / len(
                                klasters_b[j])) ** 2)


            rmin = R.min()

            # print(len(klasters_a))
            ri, rj = np.where(R == rmin)
            print(rj, ri)
            for i in range(len(ri)-1,-1,-1):
                # print(rj[i], len(klasters_a))
                klasters_a[ri[i]] = klasters_a[ri[i]] + klasters_a[rj[i]]
                klasters_b[ri[i]] = klasters_b[ri[i]] + klasters_b[rj[i]]
                klasters_b.pop(rj[i])
                klasters_a.pop(rj[i])

        print(klasters_b)
        print(klasters_a)

        amin, amax, bmin, bmax = 0, 0, 0, 0

        for i in range(len(klasters_a)):
            bmin = min(klasters_b[i])
            bmax = max(klasters_b[i])
            amin = min(klasters_a[i])
            amax = max(klasters_a[i])

            rr, cc = skimage.draw.rectangle_perimeter((amin, bmin), end=(amax, bmax), shape=frame1.shape)
            frame1[rr, cc] = (0, 255, 0)

    contours = contours.astype(np.ubyte)

    cv2.imshow("frame1", frame1)
    print("1")
    frame1 = frame2  #
    ret, frame2 = cap.read()  #

    if cv2.waitKey(40) == 27:
        break

cap.release()
cv2.destroyAllWindows()
