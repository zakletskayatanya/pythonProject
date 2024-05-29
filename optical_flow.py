import cv2
import numpy as np
import processin_image

w_size = 4
k = 0.06
p0 = {0: [], 1: [], 2: []}
circle = []
circle.append([0, -3])
circle.append([1, -3])
circle.append([2, -2])
circle.append([3, -1])
circle.append([3, 0])
circle.append([3, 1])
circle.append([2, 2])
circle.append([1, 3])
circle.append([0, 3])
circle.append([-1, 3])
circle.append([-2, 2])
circle.append([-3, 1])
circle.append([-3, 0])
circle.append([-3, -1])
circle.append([-2, -2])
circle.append([-1, 3])
t = 30
radius = 3
pixels = 16
n = 12

count = {0: 0, 1: 0, 2: 0}


def corners_fast(frame, x, y, w, h):
    points = []
    for i in range(x+radius, x + w - radius):
        for j in range(y + radius, y + h - radius):

            light_count, dark_count = 0, 0
            color_mass = np.zeros(pixels)
            for l in range(4):
                if frame[i, j] < frame[i+circle[4*l][0], j + circle[4*l][1]] - t:
                    dark_count += 1
                    continue
                if frame[i, j] > frame[i+circle[4*l][0], j + circle[4*l][1]] + t:
                    light_count += 1
                    continue
            if dark_count >= 3 or light_count >= 3:
                light_count, dark_count = 0, 0
                for k in range(pixels):
                    if frame[i, j] < frame[i+circle[k][0], j + circle[k][1]] - t:
                        color_mass[k] = -1
                        dark_count += 1
                        light_count = 0
                        continue
                    if frame[i, j] > frame[i+circle[k][0], j + circle[k][1]] + t:
                        color_mass[k] = 1
                        light_count += 1
                        dark_count = 0
                if light_count >= n or dark_count >= n:
                    points.append([i, j])
    # print(points)
    return points


def corners_fast_check(frame, points):
    for key in reversed(range(len(points))):
            i, j = points[key]
            if 3 > i > frame.shape[0]-3 or 3 > j > frame.shape[1]-3:
                points.pop(key)
                continue
            light_count, dark_count = 0, 0
            color_mass = np.zeros(pixels)
            for l in range(4):
                if frame[i, j] < frame[i+circle[4*l][0], j + circle[4*l][1]] - t:
                    dark_count += 1
                    continue
                if frame[i, j] > frame[i+circle[4*l][0], j + circle[4*l][1]] + t:
                    light_count += 1
                    continue
            if dark_count >= 3 or light_count >= 3:
                light_count, dark_count = 0, 0
                for k in range(pixels):
                    if frame[i, j] < frame[i+circle[k][0], j + circle[k][1]] - t:
                        color_mass[k] = -1
                        dark_count += 1
                        light_count = 0
                        continue
                    if frame[i, j] > frame[i+circle[k][0], j + circle[k][1]] + t:
                        color_mass[k] = 1
                        light_count += 1
                        dark_count = 0
                if light_count >= n or dark_count >= n:
                    points.append([i, j])
                else:
                    points.pop(key)
            else:
                points.pop(key)
    return points


# def corners_check(gradient_x, gradient_y, points):
#     if not points:
#         return points
#     for key in reversed(range(len(points))):
#             i, j = points[key]
#             Ix = gradient_x[i - w_size:i + w_size, j - w_size:j + w_size].flatten()
#             Iy = gradient_y[i - w_size:i + w_size, j - w_size:j + w_size].flatten()
#
#             M = np.array([[np.sum(Ix ** 2), np.sum((Ix * Iy))],
#                           [np.sum((Iy * Ix)), np.sum(Iy ** 2)]])
#             if np.linalg.det(M) == 0:
#                 points.pop(key)
#             R = np.linalg.det(M) - k * np.trace(M) ** 2
#             if R < k:
#                 points.pop(key)
#     return points


# def corners_points(gradient_x, gradient_y, x, y, w, h):
#     corners = np.zeros(gradient_x.shape)
#     for i in range(x + w_size, x + w - w_size):
#         for j in range(y + w_size, y + h - w_size):
#             Ix = gradient_x[i - w_size:i + w_size, j - w_size:j + w_size].flatten()
#             Iy = gradient_y[i - w_size:i + w_size, j - w_size:j + w_size].flatten()
#
#             M = np.array([[np.sum(Ix ** 2), np.sum((Ix * Iy))],
#                           [np.sum((Iy * Ix)), np.sum(Iy ** 2)]])
#             if np.linalg.det(M) == 0:
#                 continue
#             R = np.linalg.det(M) - k * np.trace(M) ** 2
#             if R < k:
#                 continue
#             corners[i, j] = R
#     corners = np.column_stack(np.where(corners > 0.1 * corners.max()))
#     return corners


def calculate_optical_flow(frame1, blur_image, blur_diff, treck, mask, frame_count):
    if not treck:
        return frame1
    U = []
    V = []

    height, width = frame1.shape[:2]

    for iter in range(2, -1, -1):
        frame1_pyr = cv2.resize(blur_image, (height // 2 ** iter, width // 2 ** iter), interpolation=cv2.INTER_AREA)
        frame2_pyr = cv2.resize(blur_diff, (height // 2 ** iter, width // 2 ** iter), interpolation=cv2.INTER_AREA)
        gradient_x, gradient_y = processin_image.gradient_image(frame1_pyr)
        u = np.zeros(gradient_x.shape)
        v = np.zeros(gradient_x.shape)
        if count[iter] % 2 == 0:
            # if iter == 2:
            # count[iter] += 1
            corners = []
            p0[iter].clear()
            for rect in treck:
                x, y, w, h = rect
                y = y // 2 ** iter
                x = x // 2 ** iter
                h = h // 2 ** iter
                w = w // 2 ** iter
                p = corners_fast(frame1_pyr, x, y, w, h)
                corners.extend(p)
        else:
            # if iter == 2:
            #     count[0] += 1
            p0_promej = corners_fast_check(frame1_pyr, p0[iter])
            corners = p0_promej
        corners = np.array(corners)
        # print(corners)
        for key, point in enumerate(corners):
                    i, j = point
                    Ix = gradient_x[i - w_size:i + w_size, j - w_size:j + w_size].flatten()
                    Iy = gradient_y[i - w_size:i + w_size, j - w_size:j + w_size].flatten()
                    M = np.array([[np.sum(Ix ** 2), np.sum((Ix * Iy))],
                                  [np.sum((Iy * Ix)), np.sum(Iy ** 2)]])
                    It = frame2_pyr[i - w_size:i + w_size, j - w_size:j + w_size].flatten()
                    b = -np.array([np.sum((Ix * It)), np.sum((Iy * It))])
                    uv = np.linalg.solve(M, b)
                    u[i, j] = uv[0]
                    v[i, j] = uv[1]
                    if 0 <= i + int(np.ceil(u[i, j])) < gradient_y.shape[0] and 0 <= j + int(np.ceil(v[i, j])) < gradient_y.shape[1]:
                        p0[iter].append([i + int(np.ceil(u[i, j])), j + int(np.ceil(v[i, j]))])

        U.append(u)
        V.append(v)
    u_t, v_t = np.zeros_like(U[0]), np.zeros_like(V[0])
    count[0] += 1
    count[1] += 1
    count[2] += 1
    for i in range(len(U) - 1):
        u_d = cv2.resize(U[i], (U[i].shape[1] * 2, U[i].shape[0] * 2), interpolation=cv2.INTER_AREA)
        v_d = cv2.resize(V[i], (V[i].shape[1] * 2, V[i].shape[0] * 2), interpolation=cv2.INTER_AREA)
        u_t = cv2.resize(u_t, (u_t.shape[1] * 2, u_t.shape[0] * 2), interpolation=cv2.INTER_AREA)
        v_t = cv2.resize(v_t, (v_t.shape[1] * 2, v_t.shape[0] * 2), interpolation=cv2.INTER_AREA)
        u_t += u_d + U[i + 1]
        v_t += v_d + V[i + 1]

    for corner in corners:
        i, j = corner
        # print(j,i)
        # print(int(np.ceil(i + (u_t[j, i]))), int(np.ceil(j + (v_t[j, i]))))
        mask = cv2.line(mask,(int(np.ceil((i + (u_t[i, j])))), int(np.ceil(j + (v_t[i, j])))),(i, j),
                        (0, 0, 255), 1)
        # cv2.circle(frame1, (int(np.ceil((i + (u_t[j, i])))), int(np.ceil(j + (v_t[j, i])))), 1,
        #                      (255, 0, 0), 2)
        cv2.circle(frame1, (int(np.ceil((i ))), int(np.ceil(j ))), 1,(0, 255, 0), -1)

    img = cv2.add(frame1, mask)
    count[0] += 1
    return img
