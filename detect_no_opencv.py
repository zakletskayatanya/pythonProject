import numpy as np
import cv2
import cluusters_ierarh


def detect_without_opencv(frame1,frame2, gradient_x, gradient_y):

        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_direction = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
        gradient_direction = np.round(gradient_direction / 45) * 45  # округление до красности 45

        suppressed = np.zeros_like(gradient_magnitude)
        angle = gradient_direction

        neighbors = np.zeros_like(gradient_magnitude)

        # Соседи по горизонтали
        neighbors[:, :-1] = gradient_magnitude[:, 1:]
        neighbors[:, 1:] = gradient_magnitude[:, :-1]

        # Соседи по вертикали
        neighbors[:-1, :] = np.maximum(neighbors[:-1, :], gradient_magnitude[1:, :])
        neighbors[1:, :] = np.maximum(neighbors[1:, :], gradient_magnitude[:-1, :])

        # Соседи по диагонали (все четыре направления)
        neighbors_diag1 = np.zeros_like(gradient_magnitude)
        neighbors_diag2 = np.zeros_like(gradient_magnitude)

        neighbors_diag1[:-1, :-1] = gradient_magnitude[1:, 1:]  # Сосед по диагонали влево вниз
        neighbors_diag1[1:, 1:] = np.maximum(neighbors_diag1[1:, 1:], gradient_magnitude[:-1, :-1])

        neighbors_diag2[:-1, 1:] = gradient_magnitude[1:, :-1]  # Сосед по диагонали вправо вниз
        neighbors_diag2[1:, :-1] = np.maximum(neighbors_diag2[1:, :-1], gradient_magnitude[:-1, 1:])

        mask_horizontal = np.logical_or(angle == 0, angle == 180, angle == -180)  # Горизонтальное направление
        mask_diag1 = np.logical_or(angle == 45, angle == -135)  # Диагональное направление 1
        mask_vertical = np.logical_or(angle == 90, angle == -90)  # Вертикальное направление
        mask_diag2 = np.logical_or(angle == 135, angle == -45)  # Диагональное направление 2

        max_neighbors = np.zeros_like(gradient_magnitude)
        max_neighbors[mask_horizontal] = neighbors[mask_horizontal]
        max_neighbors[mask_diag1] = neighbors[mask_diag1]
        max_neighbors[mask_vertical] = neighbors[mask_vertical]
        max_neighbors[mask_diag2] = neighbors[mask_diag2]

        suppressed = np.where(gradient_magnitude >= max_neighbors, gradient_magnitude, suppressed)

        top_threshhold = 120
        low_threshhold = 80

        suppressed = np.where(suppressed >= top_threshhold, 255, suppressed)
        suppressed = np.where(suppressed <= low_threshhold, 0, suppressed)
        suppressed = np.where((low_threshhold <= suppressed) & (suppressed <= top_threshhold),
                              100, suppressed)

        cc = cluusters_ierarh.find_clusters(suppressed)
        trecker_rect = []
        if cc is not None:
            for cluster in cc:
                contour = np.array(cluster)
                if cv2.contourArea(contour) < 50:  # условие при котором площадь выделенного объекта меньше 700 px
                    continue
                xx, yy, w, h = cv2.boundingRect(contour)
                trecker_rect.append([xx, yy, w, h])
                cv2.rectangle(frame1, (xx, yy), (xx + w, yy + h), (0, 255, 0), 2)

        return frame1, trecker_rect
