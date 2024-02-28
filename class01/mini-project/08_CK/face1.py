import cv2
import numpy as np

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 인식기 로드
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 얼굴 인식
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("얼굴을 찾을 수 없습니다.")
        return False, None

    # 가장 큰 얼굴 선택 (가정: 한 명의 사람만이 존재한다고 가정)
    (x, y, w, h) = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)[0]

    return True, gray[y:y+h, x:x+w]

def load_model_faces():
    model_faces = []
    model_face_paths = ["model1.jpg", "model2.jpg", "model3.jpg"]  # 다양한 각도의 모델 얼굴 사진 경로

    for path in model_face_paths:
        model_face = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if model_face is not None:
            model_faces.append(model_face)
        else:
            print(f"모델 얼굴을 불러오는 데 실패했습니다: {path}")

    return model_faces

def calculate_similarity(face1, face2):
    # 두 얼굴 간의 유사성 계산
    difference = cv2.absdiff(face1, face2)
    return np.mean(difference)

def main():
    # 모델 얼굴 로드
    model_faces = load_model_faces()

    if not model_faces:
        print("모델 얼굴을 찾을 수 없습니다.")
        return

    # 카메라 캡처 시작
    cap = cv2.VideoCapture(0)

    while True:
        # 카메라로부터 프레임 캡처
        ret, frame = cap.read()

        # 얼굴 인식
        face_detected, face = detect_face(frame)

        if face_detected:
            # 입력된 얼굴 영역 크기를 모델 얼굴의 크기에 맞게 조정
            face = cv2.resize(face, (model_faces[0].shape[1], model_faces[0].shape[0]))

            min_similarity = float('inf')
            for model_face in model_faces:
                # 각 모델 얼굴과의 유사성 계산
                similarity = calculate_similarity(face, model_face)
                if similarity < min_similarity:
                    min_similarity = similarity
            
            if min_similarity < 50:  # 임계값보다 작으면 같은 사람으로 판단
                print("인증되었습니다.")
                cv2.imshow('Face Detection', frame)
            else:
                # 다른 사람으로 판단하고 이미지를 캡처하여 저장
                cv2.imwrite("intruder.jpg", frame)
                print("등록되지 않은 얼굴을 감지하여 이미지를 저장했습니다.")
        
        # 결과를 화면에 표시
        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 종료
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

