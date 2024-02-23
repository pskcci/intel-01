import cv2
import numpy as np

def detect_circles(video_source=0):
    cap = cv2.VideoCapture(5)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Frame load failed!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=200, param2=30, minRadius=10, maxRadius=100)

        if circles is not None:
            circles = circles[0, :].astype(int)

            for x, y, r in circles:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow("frame", frame)

        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_circles()