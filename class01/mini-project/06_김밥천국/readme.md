![header](https://capsule-render.vercel.app/api?type=shark&color=auto&height=300&section=header&Kimbap%20Heaven=capsule%20render&fontSize=90)

# MediaPipe 와 KNN을 통해 Pose Estimation 를 사용하여 가위바위보를 검출하고 승/패자를 나타내는 프로그램
### Team: Kimbap Heaven
* Members
  | Name | Role |
  |----|----|
  | 김보람 | Project member, 프로젝트 멤버1 |
  | 천호진 | Project member, 프로젝트 멤버2 |

### 프로그램 소개
* MediaPipe 와 KNN을 통해 Pose Estimation 를 사용하여 가위바위보를 검출하고 승/패자를 나타내는 프로그램
  
### 주 활용 기술
이 프로젝트는 다음 두 가지 주요 기술을 활용합니다:
 
* MediaPipe : Google에서 개발한 오픈 소스 플랫폼 프레임워크로, 개발자들이 머신러닝을 기반으로 한 모바일 및 웹 애플리케이션에서 사용할 수 있는 다양한 미리 만들아진 솔루션을 제공

* KNN : 어떤 새로운 데이터로부터 거리가 가까운 K개의 다른 데이터의 레이블(속성)을 참고하여 K개의 데이터 중 가장 빈도 수가 높게 나온 데이터의 레이블로 분류하는 알고리즘 

### 동작 흐름

    1. 라이브러리 및 모델 초기화:
        OpenCV, Mediapipe, NumPy 등의 라이브러리를 불러옵니다.
        손 감지를 위한 mp_hands.Hands 모델과 제스처 분류를 위한 KNN (K-Nearest Neighbors) 모델을 초기화합니다.

    2. 카메라 초기화 및 화면 설정:
        OpenCV를 사용하여 카메라를 초기화하고 해상도를 설정합니다.

    3. 게임 변수 초기화:
        플레이어 1과 플레이어 2의 점수, 게임 모드, 손 감지 여부 등의 변수를 초기화합니다.

    4. 메인 루프:
        while 루프에서는 카메라에서 프레임을 읽어오고, 이미지를 전처리합니다.
        손 감지를 위해 mp_hands.Hands 모델을 사용하여 손의 랜드마크를 추출하고, 이를 기반으로 제스처 각도를 계산합니다.

    5. 제스처 분류:
        계산된 제스처 각도를 사용하여 KNN 모델을 통해 제스처를 분류하고, 화면에 결과를 표시합니다.

    6. 게임 결과 및 스코어 계산:
        두 명의 플레이어가 제스처를 내면 게임 결과를 판단하고, 승자에 따라 화면에 승리자를 표시합니다.
        게임 결과에 따라 플레이어의 점수를 증가시키고, 스코어가 10점에 도달하면 잠시 게임을 멈추고 초기화합니다.

    7. 화면 표시:
        OpenCV를 사용하여 화면에 게임 상태, 플레이어 점수, 승리 메시지 등을 표시합니다.

    8. 프로그램 종료:
        'q' 키를 누르면 프로그램이 종료됩니다.

<img src="https://capsule-render.vercel.app/api?type=shark&color=auto&height=300&section=footer&text=Kimpab%20is%20Love&fontSize=80" />
