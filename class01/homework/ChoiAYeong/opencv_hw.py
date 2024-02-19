import numpy as np
import cv2

cap = cv2.VideoCapture(3)

w = 1280
h = 720

cap.set(cv2.CAP_PROP_FRAME_WIDTH,w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

while(cap.isOpened()):
    ret, frame = cat.read()
    if ret is False:
        print("Can't receive frame.")
        break

    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 75, 150)
    cv2.imshow("edges", edges)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 130)
    if lines is not None:
        for lines is not None:
            for line in lines:
                rho, theta = line[0]
                cos, sin = np.cos(theta), np.sin(theta)
                x0, y0 = rho*cos, rho*sin
                x1, y1 = int(x0 + 1000 * (-sin)), int(y0 + 1000 * cos)
                x2, y2 = int(x0 + 1000 *sin), int(y0 +1000*(-cos))
                cv2.line(frame, (x1, y1), (x2, y2), (255,0,0), 1)
                
                
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=40, minRadius=8, maxRadius=40)
    if circles is not None:
        for i in circles[0]:
            cv2.circle(frame, (int(i[0]), int(i[1])), int(i[2]), (0,255,0),2)
            cv2.imshow("Camera", frame)
    key = cv2.waitKey(0)

    if key &0xFF == ord('q'):
        break
    
                
