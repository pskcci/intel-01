import cv2
import numpy as np

cap = cv2.VideoCapture(4)  # 웹캠 영상을 사용합니다.

while True:
    ret, frame = cap.read()
    if not ret:
        print("영상을 읽을 수 없습니다.")
        break

    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러 적용 (노이즈 제거를 위해)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # 엣지 검출 (Canny 엣지 검출)
    edges = cv2.Canny(gray_blurred, 50, 150)

    # 허프 원 변환을 이용한 원 검출
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, 
                               minDist=50, param1=100, param2=100, 
                               minRadius=10, maxRadius=80)

    # 원 검출 결과가 있으면
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # 검출된 원 표시
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), 
                                 (0, 128, 255), -1)

    # 선 검출
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # 검출된 선 표시
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 화면에 프레임 표시
    cv2.imshow("Circle and Line Detection", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC 키를 누르면 종료
        break

# 작업 완료 후 해제
cap.release()
cv2.destroyAllWindows()
