import cv2
import numpy as np

# 웹캠을 켭니다.
cap = cv2.VideoCapture(0)

# 웹캠이 정상적으로 열렸는지 확인합니다.
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# 웹캠 영상을 계속해서 표시합니다.
while True:
    ret, frame = cap.read()
    
    # 프레임을 제대로 읽었는지 확인합니다.
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break
    
    # 원 검출을 위해 이미지를 그레이스케일로 변환합니다.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 가우시안 블러를 적용하여 노이즈를 제거합니다.
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Hough Circle Transform을 적용하여 원을 검출합니다.
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=150, param2=30, minRadius=5, maxRadius=40)
    
    # 선 검출을 위한 캐니 엣지 검출기 적용
    edges = cv2.Canny(gray_blurred, 50, 150)
    
    # 검출된 선을 그릴 빈 화면 생성
    line_image = np.zeros_like(frame)
    
    # 허프 선 변환을 사용하여 선을 검출합니다.
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    
    # 검출된 선을 원본 이미지에 그립니다.
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    
    # 원이 검출되면 그리기
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # 원 주변에 원 그리기
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            # 중심점 그리기
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
    
    # 원본 프레임과 선, 엣지 검출 결과를 합칩니다.
    combined_image = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    combined_image = cv2.addWeighted(combined_image, 1, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 1, 0)
    
    # 합쳐진 이미지를 화면에 표시합니다.
    cv2.imshow("Combined Image", combined_image)
    
    # 'q' 키를 누르면 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 사용이 끝나면 웹캠을 닫고 창을 닫습니다.
cap.release()
cv2.destroyAllWindows()
