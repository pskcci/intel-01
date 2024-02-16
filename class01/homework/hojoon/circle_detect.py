import numpy as np
import cv2

cap = cv2.VideoCapture(0)

w = 640
h = 480

cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

while (cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (3, 3))

    detected_circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        1,
        50,
        param1=100,
        param2=40,
        minRadius=5,
        maxRadius=40)

    canny = cv2.Canny(gray, 75, 150)

    cv2.imshow("CANNY", canny)

    lines = cv2.HoughLines(canny, 1, np.pi / 180, 150)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            cos, sin = np.cos(theta), np.sin(theta)
            x0 = rho * cos
            y0 = rho * sin
            x1, y1 = int(x0 + 1000*(-sin)), int(y0 + 1000*(cos))
            x2, y2 = int(x0 + 1000*(sin)), int(y0 + 1000*(-cos))
            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                1)

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            cv2.circle(frame, (a, b), r, (0, 255, 0), 2)
            cv2.circle(frame, (a, b), 1, (0, 0, 255), 3)

    cv2.imshow("Detected Circle", frame)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break
