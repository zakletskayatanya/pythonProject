import random
import math
import cv2  # импорт модуля cv2
import skimage
import numpy as np
import Gaussian_filter_no_opencv as gauss
from scipy.ndimage import gaussian_filter
import map

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

    x ,y = np.where(contours == 255)
    center = np.zeros((10, 2))
    perem_a = []
    perem_b = []
    a, b = x, y
    R = np.zeros((len(a)-1, len(b)-1))
    R += 10000

    Rmin = -100
    # print(x)
    # print(a)
    klasters_a = []
    klasters_b = []
    count = 0
    while Rmin < 1:


        new_R =[]

        if count == 0:
            count += 1
            for i in range(R.shape[0]-1):
                for j in range(i+1, R.shape[1]):
                    R[i, j] = math.sqrt((a[i] - a[j+1])**2+(b[i] - b[j+1])**2)

            Rmin = R.min()
            # print(Rmin)
            for i in range(R.shape[0]-1):
                perem_a.append(a[i])
                perem_b.append(b[i])
                for j in range(i, R.shape[1]):
                    if Rmin == R[i, j]:
                        perem_a.append(a[j])
                        perem_b.append(b[j])
                        R[i, j] = 100000
                klasters_a.append(perem_a)
                klasters_b.append(perem_b)
                new_R = sum(perem_a)/ len(perem_a)
                new_R = sum(perem_b) / len(perem_b)
                perem_a = []
                perem_b = []


        if count != 0:

            print(sum(klasters_a))
            print(klasters_b)
            print(b)
            print(len(sum(klasters_a)))
            print(len(klasters_b))
            range()


        # new_a.append(a[-1])
        # a = new_a
        # new_b.append(b[-1])
        # b = new_b
        # print(a)
        print(klasters_b)
        # print(b)



    # if len(b) != 0 or len(a) != 0:
    #     bmin = []
    #     amin = []
    #     bmax = []
    #     amax = []
    #
    #     for i in range(len(center)):
    #         klaster_b = np.array(np.where(klast_b == i))
    #         klaster_a = np.array(np.where(klast_a == i))
    #
    #         # print(klaster_b.shape, klaster_a.shape)
    #         if klaster_b.shape[1] > 0:
    #             x = np.zeros(klaster_b.shape[1])
    #             y = np.zeros(klaster_a.shape[1])
    #             for j in range(klaster_a.shape[1]):
    #                 x[j] = b[klaster_b[:, j]]
    #                 y[j] = a[klaster_a[:, j]]
    #             # print(x,y)
    #             bmin.append(min(x))
    #             bmax.append(max(x))
    #             amin.append(min(y))
    #             amax.append(max(y))
    #             # print(bmin, bmax,amin,amax)
    #             for k in range(len(bmin)):
    #
    #                 rr, cc = skimage.draw.rectangle_perimeter((amin[k], bmin[k]), end=(amax[k], bmax[k]),
    #                                                       shape=frame1.shape)
    #                 frame1[rr, cc] = (0, 255, 0)

    # d_a = np.diff(a)
    # dd = np.where(d_a > 1)
    # three_a = []
    # three_b = []
    # j = 0
    # for i in range(len(dd[0])):
    #     three_a.append([1] * a[j:dd[0][i]+1])
    #     j = dd[0][i]+1
    # three_a.append([1] * a[j:])
    # # print(three_a)
    # # if len(a) != 0 or len(b) != 0:
    # #     bmin = min(b)
    # #     amin = min(a)
    # #     bmax = max(b)
    # #     amax = max(a)
    # #     rr, cc = skimage.draw.rectangle((amin, bmin), end=(amax, bmax), shape=frame1.shape)
    # #     # frame1[rr, cc] = (0, 255, 0)
    # #     n = 7
    # #
    # #     for j in range(0, len(rr)):
    # #         x = np.where(rr[j] == 255)
    # #         y = np.where(cc[j] == 255)
    # #         if len(x) != 0 and len(y) != 0:
    # #             print(x)
    # #             rr1, cc1 = skimage.draw.rectangle_perimeter((min(x), min(y)), end=(max(x), max(y)), shape=frame1.shape)
    # #             frame1[rr1, cc1] = (0, 255, 0)
    # n = 1
    # splist_a = np.array_split(a, n)
    # splist_b = np.array_split(b, n)
    #
    # if len(b) != 0 or len(a) != 0:
    #     bmin = []
    #     amin = []
    #     bmax = []
    #     amax = []
    #
    #     for j in range(0, len(splist_b)):
    #         if len(splist_b[j]) > 10 and len(splist_a[j]) > 10:
    #             # print(splist_b[j])
    #             bmin.append(min(splist_b[j]))
    #             bmax.append(max(splist_b[j]))
    #         # if len(splist_a[j] > 10):
    #             amin.append(min(splist_a[j]))
    #             amax.append(max(splist_a[j]))
    #
    #             rr, cc = skimage.draw.rectangle_perimeter((amin[j], bmin[j]), end=(amax[j], bmax[j]), shape=frame1.shape)
    #             frame1[rr, cc] = (0, 255, 0)
    #             if len(rr)*len(cc) < 1000:
    #                 pass

    contours = contours.astype(np.ubyte)

    cv2.imshow("frame1", frame1)
    print("1")
    frame1 = frame2  #
    ret, frame2 = cap.read()  #

    if cv2.waitKey(40) == 27:
        break

cap.release()
cv2.destroyAllWindows()
