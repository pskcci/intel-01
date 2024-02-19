import cv2
import numpy as np

cap = cv2.VideoCapture(4)  # 디바이스 노드 번호에 따라 인자를 조정해주세요.

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 원 검출을 위해 그레이스케일 이미지로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 원 검출을 위한 전처리 (가우시안 블러 적용)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 원 검출
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=200, param2=30, minRadius=10, maxRadius=100)
    
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    if circles is not None:
        # 검출된 원 그리기
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)

    if lines is not None:
        # 검출된 라인 그리기
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
