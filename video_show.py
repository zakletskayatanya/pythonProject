import cv2
import detect_no_opencv
import detect_with_opencv
import optical_flow
import optical_flow_opencv
import numpy as np
import processin_image


class VideoShowing:

    def __init__(self):
        self.cap = None
        # self.dim = (780//2, 480//2)

        self.dim = (540,280)
    def init(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

        cv2.namedWindow("main", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('main', self.dim)

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("File is not open")
            return None
        return frame

    def run(self):
        frame1 = self.read_frame()
        frame2 = self.read_frame()


        frame2 = cv2.resize(frame2, self.dim, interpolation=cv2.INTER_AREA)
        mask = np.zeros_like(frame2)
        mask_array = []
        frame_count=1

        while self.cap.isOpened():
            if frame2 is None or frame1 is None :
                break

            frame1 = cv2.resize(frame1, self.dim, interpolation=cv2.INTER_AREA)
            frame2 = cv2.resize(frame2, self.dim, interpolation=cv2.INTER_AREA)


            # предварительная обработка кадров
            blur_diff = processin_image.blur_diff_image(frame1, frame2)

            if frame_count % 70 == 0:
                mask_array.append(mask)
                mask = np.zeros_like(frame2)
            blur_img1 = processin_image.blur_image(frame1)
            blur_img2 = processin_image.blur_image(frame2)

            gradient_x, gradient_y = processin_image.gradient_image(blur_diff)

            # detect without opencv
            trecker_rectangle = detect_no_opencv.detect_without_opencv(frame1, frame2, gradient_x, gradient_y)

            # trecker_img = optical_flow_opencv.calcOpticalFlow(frame1, frame2, mask, trecker_rectangle, gradient_x, gradient_y)
            # detect with opencv
            # trecker_rectangle = detect_with_opencv.detect_with_opencv(frame1, frame2)

            trecker_img = optical_flow.calculate_optical_flow(frame1, blur_img1, blur_diff, trecker_rectangle, mask, frame_count)

            cv2.imshow("main", trecker_img)

            frame1 = frame2
            frame2 = self.read_frame()

            if cv2.waitKey(1) == 27:
                break
            # print(frame_count)
            frame_count += 1

        self.cap.release()
        cv2.destroyAllWindows()


