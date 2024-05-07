import cv2
import numpy as np
import processin_image

w_size = 3
k = 0.06
p0_u = np.zeros((540,280))
p0_v = np.zeros((540,280))
p0 = [[],[],[]]

def corners_points(gradient_x, gradient_y, x, y, w, h):
    corners = np.zeros(gradient_x.shape)
    for j in range(x + w_size, x + w - w_size):
        for i in range(y + w_size, y + h - w_size):
            Ix = gradient_x[i - w_size:i + w_size, j - w_size:j + w_size].flatten()
            Iy = gradient_y[i - w_size:i + w_size, j - w_size:j + w_size].flatten()

            M = np.array([[np.sum(Ix ** 2), np.sum((Ix * Iy))],
                          [np.sum((Iy * Ix)), np.sum(Iy ** 2)]])
            if np.linalg.det(M) == 0:
                continue
            R = np.linalg.det(M) - k * np.trace(M) ** 2
            if R < k:
                continue
            corners[i, j] = R
    corners = np.column_stack(np.where(corners > 0.2 * corners.max()))
    return corners


def calculate_optical_flow(frame1, blur_image, blur_diff, treck, mask, frame_count):
    U = []
    V = []

    width, height = frame1.shape[:2]

    for iter in range(2, -1, -1):
        frame1_pyr = cv2.resize(blur_image, (height // 2 ** iter, width // 2 ** iter), interpolation=cv2.INTER_AREA)
        frame2_pyr = cv2.resize(blur_diff, (height // 2 ** iter, width // 2 ** iter), interpolation=cv2.INTER_AREA)

        gradient_x, gradient_y = processin_image.gradient_image(frame1_pyr)
        u = np.zeros(gradient_x.shape)
        v = np.zeros(gradient_x.shape)
        if frame_count % 10 == 0:
            corners = []
            p0[2-iter].clear()
            for rect in treck:
                x, y, w, h = rect
                y = y // 2 ** iter
                x = x // 2 ** iter
                h = h // 2 ** iter
                w = w // 2 ** iter
                points = corners_points(gradient_x, gradient_y, x, y, w, h)
                corners.extend(points)
            p0[2-iter] = corners
        else:
            corners = p0[2 - iter]
        corners = np.array(corners)
        for key, point in enumerate(corners):
                    i, j = point
        # for i in range(corners.shape[0]):
        #         for j in range(corners.shape[1]):
                    # print(i,j)
                    Ix = gradient_x[i - w_size:i + w_size, j - w_size:j + w_size].flatten()
                    Iy = gradient_y[i - w_size:i + w_size, j - w_size:j + w_size].flatten()

                    M = np.array([[np.sum(Ix ** 2), np.sum((Ix * Iy))],
                                  [np.sum((Iy * Ix)), np.sum(Iy ** 2)]])
                    if np.linalg.det(M) == 0:
                        continue
                    It = frame2_pyr[i - w_size:i + w_size, j - w_size:j + w_size].flatten()
                    b = -np.array([np.sum((Ix * It)), np.sum((Iy * It))])
                    uv = np.linalg.solve(M, b)
                    u[i, j] = uv[0]
                    v[i, j] = uv[1]
                    if i + int(np.ceil(u[i, j])) < gradient_y.shape[0] & j + int(np.ceil(v[i, j])) < gradient_y.shape[1]:
                        p0[2-iter][key] = [i + int(np.ceil(u[i, j])), j + int(np.ceil(v[i, j]))]

            # for j in range(x + w_size, x + w - w_size):
            #     for i in range(y + w_size, y + h - w_size):
            #         Ix = gradient_x[i - w_size:i + w_size, j - w_size:j + w_size].flatten()
            #         Iy = gradient_y[i - w_size:i + w_size, j - w_size:j + w_size].flatten()
            #
            #         M = np.array([[np.sum(Ix ** 2), np.sum((Ix * Iy))],
            #                       [np.sum((Iy * Ix)), np.sum(Iy ** 2)]])
            #
            #         It = frame2_pyr[i - w_size:i + w_size, j - w_size:j + w_size].flatten()
            #         b = -np.array([np.sum((Ix * It)), np.sum((Iy * It))])
            #         uv = np.linalg.solve(M, b)
            #         u[i, j] = uv[0]
            #         v[i, j] = uv[1]
            #
            #         # if frame_count % 5 != 0:
            #         #     continue
            #         if np.linalg.det(M) == 0:
            #             continue
            #         R = np.linalg.det(M) - k * np.trace(M) ** 2
            #         if R < k:
            #             continue
            #         corners[i, j] = R

        U.append(u)
        V.append(v)
    u_t, v_t = np.zeros_like(U[0]), np.zeros_like(V[0])

    for i in range(len(U) - 1):
        u_d = cv2.resize(U[i], (U[i].shape[1] * 2, U[i].shape[0] * 2), interpolation=cv2.INTER_AREA)
        v_d = cv2.resize(V[i], (V[i].shape[1] * 2, V[i].shape[0] * 2), interpolation=cv2.INTER_AREA)
        u_t = cv2.resize(u_t, (u_t.shape[1] * 2, u_t.shape[0] * 2), interpolation=cv2.INTER_AREA)
        v_t = cv2.resize(v_t, (v_t.shape[1] * 2, v_t.shape[0] * 2), interpolation=cv2.INTER_AREA)
        u_t += u_d + U[i + 1]
        v_t += v_d + V[i + 1]

    # if frame_count > 1:
    #     p0 = np.column_stack((u_t, v_t))
    # if frame_count % 5 == 0:
    # corners = np.column_stack(np.where(corners > 0.2 * corners.max()))
    # print(np.nonzero(u_t))


    for corner in corners:
        i, j = corner
        # p0_u[j, i] = int(np.ceil(i + (u_t[j, i])))
        # p0_v[j, i] = int(np.ceil(j + (v_t[j, i])))
        # print(j,i)
        # print(int(np.ceil(i + (u_t[j, i]))), int(np.ceil(j + (v_t[j, i]))))
        mask = cv2.line(mask,  (int(np.ceil((j + (u_t[i, j])))), int(np.ceil(i + (v_t[i, j])))),(j, i),
                        (0, 255, 0), 1)
        # cv2.circle(frame1, (int(np.ceil((i + (u_t[j, i])))), int(np.ceil(j + (v_t[j, i])))), 1,
        #                      (255, 0, 0), 2)
        cv2.circle(frame1, (int(np.ceil((j + (u_t[i, j])))), int(np.ceil(i + (v_t[i, j])))), 1,(255, 0, 0), 2)

    img = cv2.add(frame1, mask)

    return img
