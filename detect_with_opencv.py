import cv2


def detect_with_opencv(frame1, frame2):
    diff = cv2.absdiff(frame1,
                       frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0, cv2.BORDER_DEFAULT)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    dilated = cv2.dilate(thresh, None,
                         iterations=3)
    сontours, _ = cv2.findContours(dilated, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    trecker_rectangle = []
    for contour in сontours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 500:
            continue
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        trecker_rectangle.append([x, y, w, h])
    return trecker_rectangle
