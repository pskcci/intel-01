import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()

    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(gray, 5000, 1500, apertureSize=5, L2gradient=True)
    lines = cv2.HoughLines(canny, 1, np.pi / 180, threshold=100)

    if lines is not None:

        for i in lines:
            rho, theta = i[0][0], i[0][1]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a*rho, b*rho

            scale = frame.shape[0] + frame.shape[1]

            x1 = int(x0 + scale * -b)
            y1 = int(y0 + scale * a)
            x2 = int(x0 - scale * -b)
            y2 = int(y0 - scale * a)

            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    else:
        print("No lines detected.")

    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1,
                               100, param1=250, param2=10, minRadius=80, maxRadius=120)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 3)

    cv2.imshow("dst", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
