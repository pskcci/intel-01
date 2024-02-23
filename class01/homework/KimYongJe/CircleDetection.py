import cv2
import numpy as np

# 웹캠에서 영상을 읽어옵니다
cap = cv2.VideoCapture(2)

while True:
    # 각 프레임을 읽어옵니다
    ret, frame = cap.read()
    output = frame.copy()

    # 그레이스케일 이미지로 변환합니다
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 5000, 3000, apertureSize=5, L2gradient=True)
    lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 90,
                            minLineLength=30, maxLineGap=100)

    # 허프 변환 원 검사를 이용하여 원을 찾습니다
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,
                               100, param1=50, param2=50, minRadius=80, maxRadius=120)

    # 원이 검출되었다면
    if circles is not None:
        # 반올림
        circles = np.round(circles[0, :]).astype("int")

        for i in circles:
            cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 5)  # 원 그리기

    # 직선이 검출되었다면
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]  # 선의 시작과 끝 점
            cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 직선 그리기

    # 결과를 출력합니다
    cv2.imshow("output", output)

    # 'q' 키를 누르면 종료합니다
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 작업이 끝나면 해제합니다
cap.release()
cv2.destroyAllWindows()
