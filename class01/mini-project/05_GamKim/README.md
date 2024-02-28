# OpenVino를 이용한 Smart Class 프로그램

### Team : GamKim


<눈 깜빡임과 행동 탐지를 이용한 수업 시간 집중도 확인 프로그램>

* Members
  | Name | Role |
  |----|----|
  | 감다공 | Project member, 프로젝트 멤버1 |
  | 김호준 | Project member, 프로젝트 멤버2 |

## OpenVino 의 Pre-trained 모델을 이용한 얼굴 인식 및 눈 깜빡임 탐지

### Face detection adas 0001

https://docs.openvino.ai/2023.3/omz_models_model_face_detection_adas_0001.html

![face detection](https://docs.openvino.ai/2023.3/_images/face-detection-adas-0001.png)

### Facial landmarks 35 adas 0002

https://docs.openvino.ai/2023.3/omz_models_model_facial_landmarks_35_adas_0002.html

![facial image](https://docs.openvino.ai/2023.3/_images/landmarks_illustration.png)

Face detection adas 0001과 연계해 Facial landmarks 탐지

### Open Closed eye 0001

https://docs.openvino.ai/2023.3/omz_models_model_open_closed_eye_0001.html

Facial Landmarks의 landmark를 이용해 Eye 부분을 Crop 후 Open Closed eye 모델을 이용해 눈 깜빡임 탐지

### Driver action recognition

https://docs.openvino.ai/2023.3/omz_models_model_driver_action_recognition_adas_0002.html

![driver_image](https://docs.openvino.ai/2023.3/_images/action-recognition-kelly.png)

눈 깜빡임과 연계해 행동 분석을 통해 휴대폰 통화 및 문자 행동 감지

## Review

- Action 탐지는 모델이 굉장히 제한적이기에 객체(Cell phone) 탐지를 이용하는 편이 훨씬 용이해보인다.

    - OpenVino 내에 있는 Object 탐지 모델은 탐지할 수 있는 객체가 많기 때문에 필요한 것을 추려서 사용해야 할 것 같다.

- 창문에서 햇빛이 들어오거나 조명 때문에 눈 깜빡임 검출에 어려움이 있었는데 Histogram 평준화를 통해 일정한 값의 사진이 Input 으로 들어가게 할 수 있었다.

- 눈 깜빡임을 이용해 집중도를 파악하기 위해 집중도 Value를 산출하는 법이나 집중을 하고 있지 않다는 기준을 잡기가 주관적이어서 어려웠던 것 같다.