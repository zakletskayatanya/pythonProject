import cv2
import numpy as np
import Gaussian_filter_no_opencv as gauss
import skimage
import scipy

def find_clusters(contours):
    a, b = np.nonzero(contours == 255)
    klasters_a = [[a[i]] for i in range(0, len(a), 20)]
    klasters_b = [[b[i]] for i in range(0, len(b), 20)]
    if len(a) > 0 and len(b) > 0:


        while True:

            clusters_a_mean = np.array(list(map(np.mean, klasters_a)))
            clusters_b_mean = np.array(list(map(np.mean, klasters_b)))

            column_st = np.column_stack((clusters_a_mean, clusters_b_mean))

            R = scipy.spatial.distance.cdist(column_st, column_st)

            np.fill_diagonal(R, np.inf)
            min_distance_index = np.argmin(R)
            min_distance = R.min()

            if min_distance >= 80:
                break

            ri, rj = np.unravel_index(min_distance_index, R.shape)
            klasters_a[ri] = np.concatenate((klasters_a[ri], klasters_a[rj]))
            klasters_b[ri] = np.concatenate((klasters_b[ri], klasters_b[rj]))
            del klasters_b[rj]
            del klasters_a[rj]
    return klasters_a, klasters_b

# Открываем видео
cap = cv2.VideoCapture("111.mp4")

# Чтение первого кадра
ret1, frame1 = cap.read()
frame1 = frame1[:-50, :]
mask = np.zeros_like(frame1)

# Параметры для оптического потока
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Создаем случайные цвета для отрисовки траектории
color = np.random.randint(0, 255, (100, 3))

# История точек для отслеживания
history_points = []

while cap.isOpened():
    # Чтение нового кадра
    ret, frame2 = cap.read()
    frame2 = frame2[:-50, :]
    mask2 = np.zeros_like(frame2)

    # Преобразование кадров в формат float32
    img_f1 = frame1.astype(np.float32)
    img_f2 = frame2.astype(np.float32)

    # Разность между кадрами
    diff_custom = np.abs(img_f1 - img_f2)
    gray_custom = cv2.cvtColor(diff_custom.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # Применение фильтра Гаусса
    blur_custom = gauss.GaussianFilter(9).gauss_blur(gray_custom)
    threshold = 40

    # Применение порога
    thresh_custom = 255 * (blur_custom > threshold)

    # Нахождение контуров
    contours1 = thresh_custom[1:, :]
    contours2 = thresh_custom[:-1, :]
    contours = 255 * (np.abs(contours1 - contours2) > 0)

    # Нахождение кластеров
    klasters_a, klasters_b = find_clusters(contours)

    # Отрисовка рамок вокруг объектов
    amin, amax, bmin, bmax = 0, 0, 0, 0
    corner_y, corner_x = np.zeros(len(klasters_a)), np.zeros(len(klasters_a))

    if len(klasters_b) != 0 and len(klasters_a) != 0:
        for i in range(len(klasters_a)):
            bmin = min(klasters_b[i])
            bmax = max(klasters_b[i])
            amin = min(klasters_a[i])
            amax = max(klasters_a[i])

            corner_x[i] = int((amax + amin) // 2)
            corner_y[i] = int((bmax + bmin) // 2)

            rr, cc = skimage.draw.rectangle_perimeter((amin, bmin), end=(amax, bmax), shape=frame1.shape)
            frame1[rr, cc] = (0, 255, 0)

    # Создание массива из координат угловых точек
        corners = np.column_stack([corner_y, corner_x]).astype(np.float32)
        corners = corners.reshape(-1, 1, 2)
        print(corners)

    # Добавление точек в историю
        history_points.append(corners)

    # Используем оптический поток для отслеживания точек
    if len(history_points) > 1:
        prev_points = np.concatenate(history_points[-2], axis=0)
        next_points, st, err = cv2.calcOpticalFlowPyrLK(
            cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY),
            prev_points,
            None,
            **lk_params
        )

        # Выбор хороших точек
        good_new = next_points[st.flatten() == 1]

        # Рисование траектории
        for i, (new, old) in enumerate(zip(good_new, prev_points)):
            a, b = new.ravel()
            c, d = old.ravel()
            if i < len(color):
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame1 = cv2.circle(frame1, (int(a), int(b)), 5, color[i % len(color)].tolist(), -1)

    # Добавление маски кадра
    img = cv2.add(frame1, mask)

    # Отображение кадра
    cv2.imshow("frame1", img)
    print("1")

    # Переключение кадров
    frame1 = frame2

    # Обработка событий клавиатуры
    if cv2.waitKey(40) == 27:
        break

# Закрытие видео
cap.release()
cv2.destroyAllWindows()