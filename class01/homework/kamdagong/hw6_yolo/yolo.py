from ultralytics import YOLO
import cv2

model = YOLO('yolov5s.pt')

im2 = cv2.imread('cat1.jpg')
results = model.predict(source=im2, save=True, save_txt=False)

if results[0].boxes.conf > 0.5:
    print("A confidence is .......")
    print(results[0].boxes.conf)
