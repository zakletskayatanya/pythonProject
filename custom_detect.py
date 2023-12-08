import cv2  # импорт модуля cv2
import skimage
import numpy as np
import Gaussian_filter_no_opencv as gauss
from scipy.ndimage import gaussian_filter

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
    threshold = 20 # переход к бинарному изображению
    thresh_custom = 255 * (blur_custom > threshold)
    # thresh_custom = thresh_custom.astype(np.ubyte)

    # contours = skimage.measure.find_contours(thresh_custom)
    contours1 = thresh_custom[:-1, :]
    contours2 = thresh_custom[1:, :]

    contours = 255 * (np.abs(contours1 - contours2) > 0)
    contours = contours.astype(np.ubyte)

    # сontours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # нахождение массива контурных точек
    # #cv2.drawContours(frame1, сontours, -1, (0, 255, 0), 2) #также можно было просто нарисовать контур объекта
    #
    # for contour in сontours:
    #      (x, y, w, h) = cv2.boundingRect(contour)  # преобразование массива из предыдущего этапа в кортеж из четырех координат
    #
    #      # метод contourArea() по заданным contour точкам, здесь кортежу, вычисляет площадь зафиксированного объекта в каждый момент времени, это можно проверить
    #      print(cv2.contourArea(contour))
    #
    #      if cv2.contourArea(contour) < 500:  # условие при котором площадь выделенного объекта меньше 700 px
    #          continue
    #      cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)  # получение прямоугольника из точек кортежа

    cv2.imshow("frame1", contours)
    print("1")
    frame1 = frame2  #
    ret, frame2 = cap.read()  #

    if cv2.waitKey(40) == 27:
        break

cap.release()
cv2.destroyAllWindows()