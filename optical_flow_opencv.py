import cv2
import numpy as np
import optical_flow


lk_params = dict( winSize = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                              10, 0.03))
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
corners = []

p0 = []



def calcOpticalFlow(frame1, frame2, mask, rectangls, gradient_x, gradient_y):

    old_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    new_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    # print(dd(old_frame))
    if not p0:
        for rect in rectangls:
            x, y, w, h = rect
            y = y
            x = x
            h = h
            w = w
            points = optical_flow.corners_points(gradient_x, gradient_y, x, y, w, h)
            corners.extend(points)
        if not corners:
            return cv2.add(frame1, mask)
        print(corners)
        for i in corners:
            # print(i)
            x, y= i
            p0.extend([np.array([x, y], dtype=np.float32)])
        # p0 = [[i] for i in corners]
        p0 = np.asarray(p0)
        print(p0)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, new_frame, p0, None, **lk_params)
    print(p1)
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new,
                                       good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)),
                        (255, 0, 0), 2)

        frame1 = cv2.circle(frame1, (int(a), int(b)), 5,
                           (0, 0, 255), -1)

    img = cv2.add(frame1, mask)
    p0 = good_new.reshape(-1, 1, 2)

    return img