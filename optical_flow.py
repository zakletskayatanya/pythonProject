import cv2
import numpy as np


class OpticalFlowProcessor:

    def __init__(self, frame1):
        self.color = np.random.randint(0, 255, (100, 3))
        self.mask = np.zeros_like(frame1)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def calculate_optical_flow(self, frame1, frame2, history_points):
        if len(history_points) > 1:
            prev_points = np.concatenate(history_points[-2], axis=0)
            next_points, st, err = cv2.calcOpticalFlowPyrLK(
                cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY),
                prev_points,
                None,
                **self.lk_params
            )

            good_new = next_points[st.flatten() == 1]

            for i, (new, old) in enumerate(zip(good_new, prev_points)):
                a, b = new.ravel()
                c, d = old.ravel()
                if i < len(self.color):
                    self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
                    frame1 = cv2.circle(frame1, (int(a), int(b)), 5, self.color[i % len(self.color)].tolist(), -1)

        img_result = cv2.add(frame1, self.mask)
        return img_result