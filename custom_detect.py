import cv2
import scipy
import skimage
import numpy as np
import Gaussian_filter_no_opencv as gauss

# cap = cv2.VideoCapture("http://192.168.217.103/mjpg/video.mjpg")
cap = cv2.VideoCapture("IMG_8546.MP4")

ret, frame1 = cap.read()
ret1, frame2 = cap.read()

while cap.isOpened():

    img_f1 = frame1.astype(np.float32)
    img_f2 = frame2.astype(np.float32)
    diff_custom = np.abs(img_f1 - img_f2)

    gray_custom = skimage.color.rgb2gray(diff_custom)

    blur_custom = gauss.GaussianFilter(7).gauss_blur(gray_custom)
    threshold = 20
    thresh_custom = 255 * (blur_custom > threshold)

    contours1 = thresh_custom[:-1, :]
    contours2 = thresh_custom[1:, :]
    contours = 255 * (np.abs(contours1 - contours2) > 0)

    a, b = np.nonzero(contours == 255)
    if len(a) > 0 and len(b) > 0:

        klasters_a = [[a[i]] for i in range(0, len(a), 20)]
        klasters_b = [[b[i]] for i in range(0, len(b), 20)]
        while True:

            clusters_a_mean = np.array(list(map(np.mean, klasters_a)))
            clusters_b_mean = np.array(list(map(np.mean, klasters_b)))

            column_st = np.column_stack((clusters_a_mean, clusters_b_mean))

            R = scipy.spatial.distance.cdist(column_st, column_st)

            np.fill_diagonal(R, np.inf)
            min_distance_index = np.argmin(R)
            min_distance = R.min()

            if min_distance >= 95:
                break

            ri, rj = np.unravel_index(min_distance_index, R.shape)
            klasters_a[ri] = np.concatenate((klasters_a[ri], klasters_a[rj]))
            klasters_b[ri] = np.concatenate((klasters_b[ri], klasters_b[rj]))
            del klasters_b[rj]
            del klasters_a[rj]

        amin, amax, bmin, bmax = 0, 0, 0, 0

        for i in range(len(klasters_a)):
            bmin = min(klasters_b[i])
            bmax = max(klasters_b[i])
            amin = min(klasters_a[i])
            amax = max(klasters_a[i])

            rr, cc = skimage.draw.rectangle_perimeter((amin, bmin), end=(amax, bmax), shape=frame1.shape)
            frame1[rr, cc] = (0, 255, 0)

    cv2.imshow("frame1", frame1)
    print("1")
    frame1 = frame2  #
    ret, frame2 = cap.read()  #

    if cv2.waitKey(40) == 27:
        break

cap.release()
cv2.destroyAllWindows()
