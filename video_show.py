import cv2
import detect_no_opencv
import detect_with_opencv
import optical_flow
import numpy as np
import gaussian_filter_no_opencv as gauss
from scipy.signal import convolve2d


class VideoShowing:

    def init(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.detect_no_opencv = detect_no_opencv.VideoProcessingWithoutOpencv()
        self.detect_opencv = detect_with_opencv.VideoProcessingWithOpencv()

        cv2.namedWindow("main", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('main', 840, 480)

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("File is not open")
            return None
        return frame

    def run(self):
        frame1 = self.read_frame()
        frame2 = self.read_frame()

        trecker = optical_flow.OpticalFlowProcessor(frame1)
        frame2 = cv2.resize(frame2, (840, 480), interpolation=cv2.INTER_AREA)
        mask = np.zeros_like(frame2)
        frame_counter = 0
        while self.cap.isOpened():
            if frame2 is None or frame1 is None:
                break

            dim = (840, 480)
            frame1 = cv2.resize(frame1, dim, interpolation=cv2.INTER_AREA)
            frame2 = cv2.resize(frame2, dim, interpolation=cv2.INTER_AREA)

            #предварительная обработка кадров
            diff = cv2.absdiff(frame1, frame2)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur_diff = gauss.GaussianFilter(7).gauss_blur(gray_diff)
            gray_img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray_img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            blur_img1 = gauss.GaussianFilter(7).gauss_blur(gray_img1)
            blur_img2 = gauss.GaussianFilter(7).gauss_blur(gray_img2)
            sobel_x = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
            sobel_y = np.array([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]])
            gradient_x_diff = convolve2d(blur_diff, sobel_x, mode='same')
            gradient_y_diff = convolve2d(blur_diff, sobel_y, mode='same')
            gradient_x = convolve2d(blur_img1, sobel_x, mode='same')
            gradient_y = convolve2d(blur_img1, sobel_y, mode='same')

            detect_img, treck, grad_x, grad_y, con, points = self.detect_no_opencv.detect_without_opencv(frame1, gradient_x_diff, gradient_y_diff)
            # if count % 3 == 0:
            trecker_img = trecker.calculate_optical_flow(frame1, blur_img1, blur_diff, treck, gradient_x, gradient_y, blur_img2, mask, frame_counter)

            cv2.imshow("main", trecker_img)
            frame1 = frame2
            frame2 = self.read_frame()
            frame_counter += 1

            if cv2.waitKey(1) == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


vid = VideoShowing()
vid.init("229-video.mp4")
# vid.init("IMG_8546.MP4")
# vid.init("111.mp4")
# vid.init("http://192.168.217.103/mjpg/video.mjpg")
vid.run()