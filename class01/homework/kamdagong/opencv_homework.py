import cv2
import numpy as np
# Read from the first camera device
cap = cv2.VideoCapture(0)
w = 640  # 1280#1920
h = 480  # 720#1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
# 성공적으로 video device 가 열렸으면 while 문 반복
while (cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Display
    image = frame.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 75, 150)
    cv2.imshow("edges", edges)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                               param1=100, param2=40,
                               minRadius=8, maxRadius=40)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 130)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            cos, sin = np.cos(theta), np.sin(theta)
            cx, cy = rho * cos, rho * sin
            x1, y1 = int(cx + 1000 * (-sin)), int(cy + 1000 * cos)
            x2, y2 = int(cx + 1000 * sin), int(cy + 1000 * (-cos))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
    if circles is not None:
        for i in circles[0]:
            cv2.circle(frame,
                       (int(i[0]), int(i[1])), int(i[2]),
                       (0, 255, 0), 2)
    cv2.imshow("Camera", frame)
    # 1 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(25)
    if key & 0xFF == ord('q'):
        break
