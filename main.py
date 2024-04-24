import video_show

if __name__ == "__main__":
    file = video_show.VideoShowing()
    file.init("test_video/229-video.mp4")
    file.run()