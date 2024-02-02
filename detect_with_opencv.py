import cv2
import numpy as np


class VideoProcessingWithOpencv:

    def __init__(self):
        self.history_points = []

    def detect_with_opencv(self, frame1, frame2):
        diff = cv2.absdiff(frame1, frame2)  # нахождение разницы двух кадров, которая проявляется лишь при изменении одного из них, т.е. с этого момента наша программа реагирует на любое движение.
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # перевод кадров в черно-белую градацию
        blur = cv2.GaussianBlur(gray, (5, 5), 0, cv2.BORDER_DEFAULT)  # фильтрация лишних контуров
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)  # метод для выделения кромки объекта белым цветом

        # делаем изображение контрастнее
        # clane = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
        #
        # lab = cv2. cvtColor(diff, cv2.COLOR_BGR2LAB)
        # l,a,b = cv2.split(lab)
        # l2 = clane.apply(l)
        # lab = cv2.merge((l2,a,b))
        # diff2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        dilated = cv2.dilate(thresh, None, iterations=3)  # данный метод противоположен методу erosion(), т.е. эрозии объекта, и расширяет выделенную на предыдущем этапе область

        сontours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # нахождение массива контурных точек

        for contour in сontours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) < 500:  # условие при котором площадь выделенного объекта меньше 700 px
                continue
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)  # получение прямоугольника из точек кортежа
        return frame1
