import cv2
import numpy as np

# 카메라를 엽니다.
cap = cv2.VideoCapture(0)  # 0은 기본 카메라를 의미합니다. 만약 다른 카메라를 사용하려면 숫자를 바꿔주세요.

while True:
    # 카메라로부터 프레임을 읽어옵니다.
    ret, frame = cap.read()

    # 프레임이 올바르게 읽어와졌는지 확인합니다.
    if not ret:
        print("카메라에서 프레임을 읽지 못했습니다. 종료합니다.")
        break

    # 회색조 이미지로 변환합니다.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 원 검출을 위한 Hough 변환을 적용합니다.
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                               param1=150, param2=40, minRadius=50, maxRadius=150)

    # 검출된 원이 있을 경우에만 그리기 작업을 수행합니다.
    if circles is not None:
        circles = circles.round().astype("int")
        for (x, y, r) in circles[0]:
            # 원 주변에 원을 그립니다.
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)

    # 선도 검출을 위해 Canny 엣지 검출을 적용합니다.
    edges = cv2.Canny(gray, 50, 150)

    # 블러를 적용하여 잡음 제거
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 적응형 이진화를 통한 이미지 이진화
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

    # 윤곽선 검출을 위해 이미지 복사본을 사용합니다.
    contours_img = thresh.copy()

    # 윤곽선 검출
    contours, _ = cv2.findContours(contours_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 윤곽선을 화면에 그립니다.
    cv2.drawContours(frame, contours, -1, (255, 0, 0), 3)

    # 선도를 이용하여 선을 그려줍니다.
    lines = cv2.HoughLinesP(edges, 1, np.pi / 150, 40, minLineLength=100, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 화면에 프레임을 표시합니다.
    cv2.imshow('Circle Detection', frame)

    # 'q' 키를 누르면 루프를 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 모든 작업이 끝나면 객체들을 해제합니다.
cap.release()
cv2.destroyAllWindows()
