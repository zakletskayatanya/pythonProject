import cv2  # импорт модуля cv2
import skimage
import math
import numpy as np
from scipy.ndimage import gaussian_filter

cap = cv2.VideoCapture("http://192.168.217.103/mjpg/video.mjpg")  # видео поток с веб камеры

#cap.set(3, 1280)  # установка размера окна
#cap.set(4, 700)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

def gauss_kernel(sigma):
    radius = round(3 * sigma)
    k = radius * 2 + 1
    kernel = np.zeros((k, k))
    y = radius
    for i in range(k):
        x = -radius
        for j in range(k):
            kernel[i, j] = (1 / (2 * math.pi * (sigma * sigma))) * math.exp(((-(x * x) - (y * y))) / (2 * (sigma * sigma)))
            x += 1
        y -= 1
    return kernel / np.sum(kernel), k, radius
def gauss_blur(img, radius):
    blur = np.sum([np.multiply(img[(i-radius):(i+radius+1), (j-radius):(j+radius+1)], kernel) for i in range(radius, img.shape[0] - radius) for j in range(radius, img.shape[1] - radius)])
    return blur

while cap.isOpened():  # метод isOpened() выводит статус видеопотока

    img_f1 = frame1.astype(np.float32) #разница 2х кадров
    img_f2 = frame2.astype(np.float32)
    diff_custom = abs(img_f1 - img_f2)
    diff_custom = diff_custom.astype(np.ubyte)
    gray_custom = skimage.color.rgb2gray(diff_custom)#черно белый кадр

    gray_custom = gray_custom.astype(np.float32)
    [kernel, k, radius] = gauss_kernel(1)
    blur_custom = gauss_blur(gray_custom, radius)
    thresh_custom = np.clip(blur_custom, 10, 255)
    # threshold = 10
    # thresh_custom = 1.0 * (blur_custom > threshold)
    # thresh_custom = thresh_custom.astype(np.ubyte)
    # делаем изображение контрастнее
    # clane = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    #
    # lab = cv2. cvtColor(diff, cv2.COLOR_BGR2LAB)
    # l,a,b = cv2.split(lab)
    # l2 = clane.apply(l)
    # lab = cv2.merge((l2,a,b))
    # diff2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    cv2.imshow("2", thresh_custom)

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

    cv2.imshow("frame1", frame1)
    print("1")
    frame1 = frame2  #
    ret, frame2 = cap.read()  #

    if cv2.waitKey(40) == 27:
        break

cap.release()
cv2.destroyAllWindows()
