import cv2
import detect_no_opencv
import detect_with_opencv
import optical_flow


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
        while self.cap.isOpened():
            if frame2 is None:
                break
            if frame1 is None:
                break

            dim = (840, 480)
            frame1 = cv2.resize(frame1, dim, interpolation=cv2.INTER_AREA)
            frame2 = cv2.resize(frame2, dim, interpolation=cv2.INTER_AREA)

            detect_img, treck, grad_x, grad_y, grad_time, con, points = self.detect_no_opencv.detect_without_opencv(frame1, frame2)

            trecker_img, img = trecker.calculate_optical_flow(frame1, frame2, treck, grad_x, grad_y, grad_time, points)
            # print(trecker_img)
            # detect_img1 = self.detect_opencv.detect_with_opencv(frame1, frame2)

            cv2.imshow("main", trecker_img)
            frame1 = frame2
            frame2 = self.read_frame()

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