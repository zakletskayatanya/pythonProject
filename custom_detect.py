import math

import cv2
import scipy
import skimage
import numpy as np
import Gaussian_filter_no_opencv as gauss

# cap = cv2.VideoCapture("http://192.168.217.103/mjpg/video.mjpg")
cap = cv2.VideoCapture("IMG_8546.MP4")

ret1, frame2 = cap.read()
frame2 = frame2[:-50,:]
mask = np.zeros_like(frame2)

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )  # Тип детектора (9/16 точек)
old_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
print(p0, type(p0))
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = (np.random.randint(0, 255, (100, 3)))

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

            if min_distance >= 100:
                break

            ri, rj = np.unravel_index(min_distance_index, R.shape)
            klasters_a[ri] = np.concatenate((klasters_a[ri], klasters_a[rj]))
            klasters_b[ri] = np.concatenate((klasters_b[ri], klasters_b[rj]))
            del klasters_b[rj]
            del klasters_a[rj]
    return klasters_a, klasters_b

def p_0(klas_a, klast_b):
    pa = np.round(np.array(list(map(np.mean, klas_a))))
    pb = np.round(np.array(list(map(np.mean, klast_b))))
    return np.column_stack([pb, pa]).astype(np.float32)
count = 0
while cap.isOpened():

    ret, frame1 = cap.read()
    frame1 = frame1[:-50, :]
    mask2 = np.zeros_like(frame1)

    img_f1 = frame1.astype(np.float32)
    img_f2 = frame2.astype(np.float32)
    diff_custom = np.abs(img_f1 - img_f2)

    gray_custom = skimage.color.rgb2gray(diff_custom)

    blur_custom = gauss.GaussianFilter(9).gauss_blur(gray_custom)
    threshold = 40
    thresh_custom = 255 * (blur_custom > threshold)

    contours1 = thresh_custom[1:, :]
    contours2 = thresh_custom[:-1, :]
    contours = 255 * (np.abs(contours1 - contours2) > 0)

    klasters_a, klasters_b = find_clusters(contours)

    amin, amax, bmin, bmax = 0, 0, 0, 0
    corner_y, corner_x = np.zeros(len(klasters_a)), np.zeros(len(klasters_a))
    if len(klasters_b) != 0 and len(klasters_a) != 0:
        for i in range(len(klasters_a)):

            bmin = min(klasters_b[i])
            bmax = max(klasters_b[i])
            amin = min(klasters_a[i])
            amax = max(klasters_a[i])

            corner_x[i] = int((amax + amin) / 2)
            corner_y[i] = int((bmax + bmin) / 2)

            rr, cc = skimage.draw.rectangle_perimeter((amin, bmin), end=(amax, bmax), shape=frame1.shape)
            frame1[rr, cc] = (0, 255, 0)


    # p0 = p_0(klasters_a, klasters_b)
    # corner_x = corner_x.reshape(-1, 1)
    # corner_y = corner_y.reshape(-1, 1)
    #
    # p0 = np.dstack([corner_y, corner_x]).astype(np.float32)
    if count == 0:
        p0 = np.dstack([corner_y[:, np.newaxis], corner_x[:, np.newaxis]]).astype(np.float32)
    # print(p0, p0.shape)
    p1, st, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), p0, None, **lk_params)
    # Select good points
    if p1 is not None:
        good_new = p1[st.flatten() == 1]
        good_old = p0[st.flatten() == 1]
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        if i < len(color):
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame1 = cv2.circle(frame1, (int(a), int(b)), 5, color[i % len(color)].tolist(), -1)
    img = cv2.add(frame1, mask)


    cv2.imshow("frame1", img)
    print("1")
    # frame1 = frame2  #
    # ret, frame2 = cap.read()  #
    p0 = good_new.reshape(-1, 1, 2)

    if cv2.waitKey(40) == 27:
        break
    count += 1

cap.release()
cv2.destroyAllWindows()
