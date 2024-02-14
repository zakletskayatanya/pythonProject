import cv2
import numpy as np


class OpticalFlowProcessor:

    def __init__(self, frame1):
        self.history_points = []
        self.color = (0, 0, 255)
        self.mask = np.zeros_like(frame1)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def calculate_optical_flow(self, frame1, frame2, hist_points):
        if len(hist_points) > 1:
            self.history_points = hist_points
            print(len(hist_points))
            prev_points = np.concatenate(self.history_points[-2], axis=0)
            next_points, st, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY),
                                                            cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY),
                                                            prev_points,
                                                            None,
                                                            **self.lk_params
                                                            )

            good_new = next_points[st.flatten() == 1]
            good_old = prev_points[st.flatten() == 1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                # if i < len(self.color):
                self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color, 2)
                frame1 = cv2.circle(frame1, (int(a), int(b)), 5, self.color, -1)

        img_result = cv2.add(frame1, self.mask)
        return img_result
