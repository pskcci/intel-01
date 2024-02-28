import urllib.request
urllib.request.urlretrieve(
    url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
    filename='notebook_utils.py'
)
import collections
import tarfile
import time
from pathlib import Path
import os
import cv2
import numpy as np
from IPython import display
import openvino as ov
from openvino.tools.mo.front import tf as ov_tf_front
from openvino.tools import mo
from decoder import OpenPoseDecoder
import sys
sys.path.append("../utils")
import notebook_utils as utils
import ipywidgets as widgets
from numpy.lib.stride_tricks import as_strided

# 모델을 다운로드할 기본 디렉토리 설정
base_model_dir = Path("model")

# Open Model Zoo에서 사용할 모델의 이름 설정
model_name = "ssdlite_mobilenet_v2"

# 모델 파일명과 다운로드 URL 설정
archive_name = Path(f"{model_name}_coco_2018_05_09.tar.gz")
model_url = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/{model_name}/{archive_name}"

# 모델 아카이브 다운로드
downloaded_model_path = base_model_dir / archive_name
if not downloaded_model_path.exists():
    utils.download_file(model_url, downloaded_model_path.name, downloaded_model_path.parent)

# 모델 압축 해제
tf_model_path = base_model_dir / archive_name.with_suffix("").stem / "frozen_inference_graph.pb"
if not tf_model_path.exists():
    with tarfile.open(downloaded_model_path) as file:
        file.extractall(base_model_dir)

# 변환할 모델의 정밀도 설정
precision = "FP16"

# 변환된 모델을 저장할 경로 설정
converted_model_path = Path("model") / f"{model_name}_{precision.lower()}.xml"

# 모델을 IR(Intel's Intermediate Representation) 형태로 변환 (이미 변환된 경우 생략)
trans_config_path = Path(ov_tf_front.__file__).parent / "ssd_v2_support.json"
if not converted_model_path.exists():
    ov_model = mo.convert_model(
        tf_model_path, 
        compress_to_fp16=(precision == 'FP16'), 
        transformations_config=trans_config_path,
        tensorflow_object_detection_api_pipeline_config=tf_model_path.parent / "pipeline.config", 
        reverse_input_channels=True
    )
    ov.save_model(ov_model, converted_model_path)
    del ov_model

# 두 번째 모델에 대한 설정
model_name2 = "human-pose-estimation-0001"
model_path = base_model_dir / "intel" / model_name2 / precision / f"{model_name2}.xml"

# 두 번째 모델 다운로드 (이미 존재하지 않는 경우)
if not model_path.exists():
    model_url_dir = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/{model_name2}/{precision}/"
    utils.download_file(model_url_dir + model_name2 + '.xml', model_path.name, model_path.parent)
    utils.download_file(model_url_dir + model_name2 + '.bin', model_path.with_suffix('.bin').name, model_path.parent)

# OpenVINO Core 인스턴스 생성
core = ov.Core()

# 디바이스 선택 위젯 생성 (사용 가능한 디바이스 목록 표시)
device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='디바이스:',
    disabled=False,
)

# 첫 번째 모델을 읽고 컴파일
model = core.read_model(model=converted_model_path)
compiled_model = core.compile_model(model=model, device_name=device.value)

# 입력 및 출력 노드 정보 획득
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# 입력 크기 정보 획득
height, width = list(input_layer.shape)[1:3]

# COCO 데이터셋 클래스 정보 및 'cell phone' 클래스의 인덱스 획득
classes = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet",
    "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush", "hair brush"
]
cellphone_index = classes.index("cell phone")
print(f"'cell phone'의 인덱스: {cellphone_index}")

# 클래스별 색상 설정
colors = cv2.applyColorMap(
    src=np.arange(0, 255, 255 / len(classes), dtype=np.float32).astype(np.uint8),
    colormap=cv2.COLORMAP_RAINBOW,
).squeeze()

# 두 번째 모델을 읽고 컴파일
model2 = core.read_model(model_path)
compiled_model2 = core.compile_model(model=model2, device_name=device.value, config={"PERFORMANCE_HINT": "LATENCY"})

# 입력 및 출력 노드 정보 획득
input_layer2 = compiled_model2.input(0)
output_layers2 = compiled_model2.outputs

# 입력 크기 정보 획득
height2, width2 = list(input_layer2.shape)[2:]

# 포즈 추정 디코더 초기화 (가정: OpenPoseDecoder가 정의되어 있음)
decoder = OpenPoseDecoder()

colors2 = ((255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85), (255, 0, 170), (85, 255, 0),
          (255, 170, 0), (0, 255, 0), (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
          (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255), (0, 170, 255))

default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
                    (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

def process_results(frame, results, thresh=0.6):
    h, w = frame.shape[:2]  # 원본 프레임의 크기
    results = results.squeeze()
    boxes, labels, scores = [], [], []
    
    for _, label, score, xmin, ymin, xmax, ymax in results:
        if score < thresh:  # 스코어 임계값 적용
            continue
        boxes.append((int(xmin * w), int(ymin * h), int((xmax - xmin) * w), int((ymax - ymin) * h)))
        labels.append(int(label))
        scores.append(float(score))

    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=thresh, nms_threshold=0.4)  # nms_threshold 조정
    
    return [(labels[idx], scores[idx], boxes[idx]) for idx in indices.flatten()] if len(indices) > 0 else []

def draw_boxes(frame, boxes):
    for label, score, box in boxes:
        color = tuple(map(int, colors[label]))
        x2, y2 = box[0] + box[2], box[1] + box[3]
        cv2.rectangle(frame, box[:2], (x2, y2), color, 3)
        
        cv2.putText(frame, f"{classes[label]} {score:.2f}", (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1, cv2.LINE_AA)  # fontScale 조정

    return frame

def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    """
    2D 풀링을 수행합니다.

    매개변수:
    - A: 입력 2D 배열
    - kernel_size: 윈도우의 크기
    - stride: 윈도우의 스트라이드
    - padding: 입력의 양쪽에 추가할 패딩의 크기
    - pool_mode: 풀링 모드 ('max' 또는 'avg')
    """
    A_padded = np.pad(A, padding, mode='constant')
    output_shape = ((A_padded.shape[0] - kernel_size) // stride + 1,
                    (A_padded.shape[1] - kernel_size) // stride + 1)
    A_w = as_strided(A_padded,
                     shape=output_shape + (kernel_size, kernel_size),
                     strides=(stride * A_padded.strides[0], stride * A_padded.strides[1]) + A_padded.strides)
    A_w = A_w.reshape(-1, kernel_size, kernel_size)

    if pool_mode == 'max':
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1, 2)).reshape(output_shape)

def heatmap_nms(heatmaps, pooled_heatmaps):
    """
    히트맵에서 Non-Maximum Suppression을 수행합니다.

    매개변수:
    - heatmaps: 원본 히트맵
    - pooled_heatmaps: 풀링된 히트맵
    """
    return heatmaps * (heatmaps == pooled_heatmaps)

def process_results2(img, pafs, heatmaps):
    """
    포즈 추정 결과를 처리합니다.

    매개변수:
    - img: 원본 이미지
    - pafs: Part Affinity Fields
    - heatmaps: 히트맵
    """
    pooled_heatmaps = np.array([[pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode='max') for h in heatmaps[0]]])
    nms_heatmaps = heatmap_nms(heatmaps, pooled_heatmaps)

    # 디코더를 사용한 포즈 해석 (decoder가 미리 정의되어 있어야 함)
    poses, scores = decoder(heatmaps, nms_heatmaps, pafs)
    output_shape = list(compiled_model2.output(index=0).partial_shape)
    output_scale = img.shape[1] / output_shape[3].get_length(), img.shape[0] / output_shape[2].get_length()

    poses[:, :, :2] *= output_scale
    return poses, scores

def draw_poses(img, poses, point_score_threshold, skeleton=default_skeleton):
    if poses.size == 0:
        return img

    for pose in poses:
        points = pose[:, :2].astype(np.int32)
        points_scores = pose[:, 2]
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img, tuple(p), 1, colors2[i], 2)
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                cv2.line(img, tuple(points[i]), tuple(points[j]), color=colors2[j], thickness=4)
    return cv2.addWeighted(img, 0.4, img, 0.6, 0)


def run_object_detection(source=0, flip=False, use_popup=False, skip_first_frames=0):
    pafs_output_key = compiled_model2.output("Mconv7_stage2_L1")
    heatmaps_output_key = compiled_model2.output("Mconv7_stage2_L2")
    player = None
    try:
        player = utils.VideoPlayer(source, flip=flip, fps=30, skip_first_frames=skip_first_frames)
        player.start()
        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        processing_times = collections.deque()
        while True:
            frame = player.next()
            if frame is None:
                print("Source ended")
                break
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            input_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)[np.newaxis, ...]
            input_img2 = cv2.resize(frame, (width2, height2), interpolation=cv2.INTER_AREA).transpose((2, 0, 1))[np.newaxis, ...]
            start_time = time.time()
            results = compiled_model([input_img])[output_layer]
            results2 = compiled_model2([input_img2])
            stop_time = time.time()
            pafs = results2[pafs_output_key]
            heatmaps = results2[heatmaps_output_key]
            poses, scores = process_results2(frame, pafs, heatmaps)
            boxes = process_results(frame=frame, results=results)
            processing_times.append(stop_time - start_time)
            if len(processing_times) > 200:
                processing_times.popleft()
            _, f_width = frame.shape[:2]
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(
                img=frame,
                text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=f_width / 1000,
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            for label, _, box in boxes:
                if label == 77:
                    # 휴대폰의 중심점 좌표
                    y = (box[1]+box[3]/2)
                    x = (box[0] + box[2]/ 2) 
                    print (poses[-1][0][0], poses[-1][0][1])
                    if poses.shape[0] !=0:
                        # 사람의 코의 위치를 point로 표현
                        point = (int(poses[-1][0][0]), int(poses[-1][0][1]))
                        cell_phone_pose = (int(x), int(y))
                        cv2.circle(frame, point, 1, (255, 0, 0), 2)
                        cv2.circle(frame, cell_phone_pose, 1, (0, 0, 255), 2)
                        if poses[-1][0][1] >= y:
                            cv2.putText(frame, "Danger", (100, 100), cv2.FONT_HERSHEY_COMPLEX, f_width / 1000, (0, 0, 255), 1, cv2.LINE_AA)
            if use_popup:
                cv2.imshow(winname=title, mat=frame)
                key = cv2.waitKey(1)
                if key == 27:
                    break
            else:
                _, encoded_img = cv2.imencode(ext=".jpg", img=draw_poses(frame, poses, 0.6), params=[cv2.IMWRITE_JPEG_QUALITY, 100])
                i = display.Image(data=encoded_img)
                display.clear_output(wait=True)
                display.display(i)
    except KeyboardInterrupt:
        print("Interrupted")
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            player.stop()
        if use_popup:
            cv2.destroyAllWindows()

USE_WEBCAM = False

video_file = "http://192.168.0.136:8080/stream"
source = 4 if USE_WEBCAM else video_file

run_object_detection(source=source, flip=isinstance(source, int), use_popup=True)
    