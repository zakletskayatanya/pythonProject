import cv2
import numpy as np
import processin_image


w_size = 3
k = 0.06


def calculate_optical_flow(frame11, frame1, frame2, treck, mask):

    U = []
    V = []

    for iter in range(3, -1, -1):
        frame1_pyr = cv2.resize(frame1, (840 // 2 ** iter, 480 // 2 ** iter), interpolation=cv2.INTER_AREA)
        frame2_pyr = cv2.resize(frame2, (840 // 2 ** iter, 480 // 2 ** iter), interpolation=cv2.INTER_AREA)

        gradient_x, gradient_y = processin_image.gradient_image(frame1_pyr)
        u = np.zeros(gradient_x.shape)
        v = np.zeros(gradient_x.shape)
        ugol = np.zeros(gradient_x.shape)
        for rect in treck:
            x, y, w, h = rect
            y = y // 2 ** iter
            x = x // 2 ** iter
            h = h // 2 ** iter
            w = w // 2 ** iter
            for j in range(x + w_size, x + w - w_size):
                for i in range(y + w_size, y + h - w_size):
                    Ix1 = gradient_x[i - w_size:i + w_size, j - w_size:j + w_size].flatten()
                    Iy1 = gradient_y[i - w_size:i + w_size, j - w_size:j + w_size].flatten()

                    M = np.array([[np.sum(Ix1 ** 2), np.sum((Ix1 * Iy1))],
                                  [np.sum((Iy1 * Ix1)), np.sum(Iy1 ** 2)]])
                    if np.linalg.det(M) == 0:
                        continue
                    R = np.linalg.det(M) - k * np.trace(M) ** 2
                    if R < k:
                        continue
                    It = frame2_pyr[i - w_size:i + w_size, j - w_size:j + w_size].flatten()
                    b = -np.array([np.sum((Ix1 * It)), np.sum((Iy1 * It))])
                    ugol[i, j] = R
                    uv = np.linalg.solve(M, b)
                    u[i, j] = uv[0]
                    v[i, j] = uv[1]

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

    ug = np.column_stack(np.where(ugol > 0.2 * ugol.max()))
    for p in ug:
        j, i = p
            # print(p)
            # print(u_t[j, i],v_t[j, i])
            # print(math.ceil(((u_t[j, i]))), math.ceil(((v_t[j, i]))))
        mask = cv2.line(mask, (int(np.ceil((i + (u_t[j, i])))), int(np.ceil(j + (v_t[j, i])))), (i, j),
                            (0, 255, 0), 1)
        frame11 = cv2.circle(frame11, (int(np.ceil((i + (u_t[j, i])))), int(np.ceil(j + (v_t[j, i])))), 1,
                                 (255, 0, 0), 2)
    img = cv2.add(frame11, mask)

    return img
