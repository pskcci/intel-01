import os
import cv2
import platform
import argparse
import numpy as np
import matplotlib.pyplot as plt

from typing import Union
from openvino.runtime import Core

MODEL_BASE_PATH: str = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "models")

DETECT_MODEL_PATH: str = os.path.join(MODEL_BASE_PATH, "detect_model.xml")
RECOG_MODEL_PATH_1: str = os.path.join(MODEL_BASE_PATH, "recog_model_1.xml")
RECOG_MODEL_PATH_2: str = os.path.join(MODEL_BASE_PATH, "recog_model_2.xml")
RECOG_MODEL_PATH_3: str = os.path.join(MODEL_BASE_PATH, "recog_model_3.xml")

IMAGE_PATH: str = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "images")

ID: int = 4
CAM_WIDTH: int = 640
CAM_HEIGHT: int = 360
FPS: int = 60


def preprocess(image: np.ndarray, width: int, height: int, model_name: str = "arcface") -> np.ndarray:
    """
        Preprocess the image file to prepare for inference
    """
    if model_name == "facenet":
        image = cv2.resize(src=image, dsize=(width, height),
                           interpolation=cv2.INTER_AREA)
    else:
        image = cv2.resize(src=image, dsize=(width, height),
                           interpolation=cv2.INTER_AREA).transpose(2, 0, 1)

    return np.expand_dims(image, axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
        Compute the cosine similarity between two vectors
    """
    return np.dot(a, b.reshape(-1, 1)) / (np.linalg.norm(a) * np.linalg.norm(b))


def setup(target: str, model_path: str) -> tuple:
    """
        Helper fucntion to setup the OpenVINO Model
    """
    ie = Core()
    model = ie.read_model(model=model_path)
    model = ie.compile_model(model=model, device_name=target)

    input_layer = next(iter(model.inputs))
    output_layer = next(iter(model.outputs))

    return model, input_layer, output_layer, \
        (input_layer.shape[0], input_layer.shape[1],
         input_layer.shape[2], input_layer.shape[3])


def detect_faces(
    model,
    output_layer,
    image: np.ndarray,
    w: int,
    h: int,
    threshold: float = 0.9,
) -> tuple:
    """
        Detect faces in the image. Returns a tuple of label indexes, probabilities and bounding boxes. (Possibly switch to detect only single face)
    """
    result = model(inputs=[image])[output_layer].squeeze()

    label_indexes: list = []
    probs: list = []
    boxes: list = []

    if result[0][0] == -1:
        return 0, None, None
    else:
        for i in range(result.shape[0]):
            if result[i][0] == -1:
                break
            elif result[i][2] > threshold:
                label_indexes.append(int(result[i][1]))
                probs.append(result[i][2])
                boxes.append([int(result[i][3] * w),
                              int(result[i][4] * h),
                              int(result[i][5] * w),
                              int(result[i][6] * h)])
            else:
                pass
    label_indexes, probs, boxes
    return label_indexes, probs, boxes


def show_images(
    image_1: np.ndarray,
    image_2: np.ndarray,
    cmap_1: str = "gnuplot2",
    cmap_2: str = "gnuplot2",
    title_1: Union[str, None] = None,
    title_2: Union[str, None] = None,
    title: Union[str, None] = None,
) -> None:
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(src=image_1, code=cv2.COLOR_BGR2RGB), cmap_1)
    plt.axis("off")
    if title_1:
        plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(src=image_2, code=cv2.COLOR_BGR2RGB), cmap_2)
    plt.axis("off")
    if title_2:
        plt.title(title_2)
    if title:
        plt.suptitle(title, fontsize=28)
    plt.show()


def faceCheckings():

    # CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", type=str,
                        default="image", help="Image/Image or Image/Realtime")
    parser.add_argument("--filename1", "-f1", type=str,
                        default="image1.jpg", help="Image Filename 1")
    parser.add_argument("--filename2", "-f2", type=str,
                        default="image2.jpg", help="Image Filename 2")
    parser.add_argument("--target", "-t", type=str,
                        default="CPU", help="Target Device for Inference")
    parser.add_argument("--model", "-mo", type=str, default="facenet",
                        help="Model to Use (arcface, facenet, sphereface)")
    args = parser.parse_args()

    # Checks for CLI arguments
    assert args.mode == "image" or args.mode == "realtime", "Invalid Mode"
    assert args.filename1 in os.listdir(IMAGE_PATH), "File 1 not Found"
    assert args.target in ["CPU", "GPU"], "Invalid Target Device"
    assert args.model == "arcface" or args.model == "facenet" or args.model == "sphereface", "Invalid Model"

    # Adaptive Equalization Setup
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(5, 5))

    # Read Reference Image and Apply CLAHE
    image = cv2.imread(os.path.join(
        IMAGE_PATH, args.filename1), cv2.IMREAD_COLOR)
    for i in range(3):
        image[:, :, i] = clahe.apply(image[:, :, i])
    temp_image = image.copy()
    h, w, _ = image.shape

    # image2 = cv2.imread(os.path.join(
    #     IMAGE_PATH, args.filename2), cv2.IMREAD_COLOR)
    # for i in range(3):
    #     image2[:, :, i] = clahe.apply(image2[:, :, i])
    # temp_image2 = image2.copy()
    # h2, w2, _ = image2.shape

    # Initialize Facial Detection Model
    d_model, _, d_output_layer, (_, _, d_H, d_W) = setup(
        args.target, DETECT_MODEL_PATH)

    # Initialize Facial Recognition Model (Facial Embeddings)
    if args.model == "arcface":
        r_model, _, r_output_layer, (_, _, r_H, r_W) = setup(
            args.target, RECOG_MODEL_PATH_1)
    elif args.model == "facenet":
        r_model, _, r_output_layer, (_, r_H, r_W, _) = setup(
            args.target, RECOG_MODEL_PATH_2)
    elif args.model == "sphereface":
        r_model, _, r_output_layer, (_, _, r_H, r_W) = setup(
            args.target, RECOG_MODEL_PATH_3)

    # Preprocess Image and Detect Faces
    image = preprocess(image, d_W, d_H)
    _, _, boxes = detect_faces(d_model, d_output_layer, image, w, h)

    # image2 = preprocess(image2, d_W, d_H)
    # _, _, boxes2 = detect_faces(d_model, d_output_layer, image2, w2, h2)

    # Preprocess face ROI Image and get embeddings
    face_image = preprocess(
        temp_image[boxes[0][1]:boxes[0][3], boxes[0][0]:boxes[0][2], :], r_W, r_H, args.model)
    reference_embeddings = r_model(inputs=[face_image])[r_output_layer]

    # face_image2 = preprocess(
    #     temp_image2[boxes2[0][1]:boxes2[0][3], boxes2[0][0]:boxes2[0][2], :], r_W, r_H, args.model)
    # reference_embeddings2 = r_model(inputs=[face_image2])[r_output_layer]

    del temp_image, boxes, image

    # del temp_image2, boxes2, image2
    # Compute Cosine Similarity between embeddings

    embeddings = r_model(inputs=[face_image])[r_output_layer]
    # embeddings2 = r_model(inputs=[face_image2])[r_output_layer]

    cs = cosine_similarity(reference_embeddings, embeddings)[0][0]
    # cs2 = cosine_similarity(reference_embeddings2, embeddings2)[0][0]

    if args.mode == "realtime":
        # Initialize Video Capture Object
        if platform.system() != "Windows":
            cap = cv2.VideoCapture(ID)
        else:
            cap = cv2.VideoCapture(ID, cv2.CAP_DSHOW)

        # Set parameters of capture object
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)

        # Read data from Video Capture Object
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Make a copy for processing purposes
            temp_frame = frame.copy()
            temp_frame2 = frame.copy()

            # Make a copy for display purposes
            disp_frame = frame.copy()

            # Apply CLAHE
            for i in range(3):
                frame[:, :, i] = clahe.apply(frame[:, :, i])
                temp_frame[:, :, i] = clahe.apply(temp_frame[:, :, i])
                # temp_frame2[:, :, i] = clahe.apply(temp_frame[:, :, i])

                # # CLAHE Display Frame (Test Purposes)
                # disp_frame[:, :, i] = clahe.apply(disp_frame[:, :, i])

            # boxes = detect_faces(
            #     d_model, d_output_layer, frame, CAM_WIDTH, CAM_HEIGHT)
            # print(boxes)
            # _, _, box = boxes
            # if not box: continue

            # for _, _, box in boxes:
            #     if len(box) != 0:
            #         face_frame = temp_frame[box[1]:box[3], box[0]:box[2], :]
            #     else:
            #         face_frame = temp_frame
            #     if face_frame.shape[0] >= 16 and face_frame.shape[1] >= 16:
            #         face_frame = preprocess(face_frame, r_W, r_H, args.model)
            #         embeddings = r_model(inputs=[face_frame])[r_output_layer]
            #         cs = cosine_similarity(reference_embeddings, embeddings)[0][0]
            #     if cs > 0.7:
            #         pt1 = (box[0], box[1])
            #         pt2 = (box[2], box[3])
            #         cv2.rectangle(disp_frame, pt1, pt2, color=(0, 255, 0))
            #         cv2.putText(disp_frame, f"{cs:.2f}", org=(
            #             box[0] + 5, box[1] + 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1, color=(0, 255, 0))

            # Preprocess Frame and Detect Faces
            frame = preprocess(frame, d_W, d_H)
            _, _, boxes = detect_faces(
                d_model, d_output_layer, frame, CAM_WIDTH, CAM_HEIGHT)
            # _, _, boxes2 = detect_faces(
            #     d_model, d_output_layer, frame, CAM_WIDTH, CAM_HEIGHT)

            # Preprocess face ROI Frame and get embeddings
            if len(boxes) != 0:
                face_frame = temp_frame[boxes[0][1]
                    :boxes[0][3], boxes[0][0]:boxes[0][2], :]
            else:
                face_frame = temp_frame

            # if len(boxes2) != 0:
            #     face_frame2 = temp_frame2[boxes2[0][1]
            #         :boxes2[0][3], boxes2[0][0]:boxes2[0][2], :]
            # else:
            #     face_frame2 = temp_frame2

            # If condition is met, compute Cosine Similarity between embeddings
            if face_frame.shape[0] < 16 or face_frame.shape[1] < 16 or face_frame2.shape[0] < 16 or face_frame2.shape[1] < 16:
                # print("ROI to small to detect")
                cv2.putText(disp_frame, "ROI to small to detect", org=(
                    25, 75), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1, color=(0, 0, 255))
            if face_frame.shape[0] >= 16 and face_frame.shape[1] >= 16:
                face_frame = preprocess(face_frame, r_W, r_H, args.model)
                embeddings = r_model(inputs=[face_frame])[r_output_layer]
                cs = cosine_similarity(reference_embeddings, embeddings)[0][0]
            # if face_frame2.shape[0] >= 16 and face_frame2.shape[1] >= 16:
            #     face_frame2 = preprocess(face_frame2, r_W, r_H, args.model)
            #     embeddings2 = r_model(inputs=[face_frame2])[r_output_layer]
            #     cs2 = cosine_similarity(
            #         reference_embeddings2, embeddings2)[0][0]
            cv2.imshow("Feed", disp_frame)

            if cs > 0.5:
                pt1 = (boxes[0][0], boxes[0][1])
                pt2 = (boxes[0][2], boxes[0][3])
                cv2.rectangle(disp_frame, pt1, pt2, color=(0, 255, 0))
                cv2.putText(disp_frame, f"{cs:.2f}", org=(
                    boxes[0][0] + 5, boxes[0][1] + 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1, color=(0, 255, 0))
                if cs > 0.7:
                    cimage = disp_frame[boxes[0][1]:boxes[0][3], boxes[0][0]:boxes[0][2]]
                    cv2.imwrite("person1.jpg", cimage)
                    break
            # if cs2 > 0.5:
            #     pt3 = (boxes2[0][0], boxes2[0][1])
            #     pt4 = (boxes2[0][2], boxes2[0][3])
            #     cv2.rectangle(disp_frame, pt3, pt4, color=(0, 255, 0))
            #     cv2.putText(disp_frame, f"{cs2:.2f}", org=(
            #         boxes2[0][0] + 5, boxes2[0][1] + 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1, color=(0, 255, 0))
            #     if cs2 > 0.7:
            #         cimage = disp_frame[boxes2[0][1]:boxes2[0][3], boxes2[0][0]:boxes2[0][2]]
            #         cv2.imwrite("person2.jpg", cimage)
            #         break
            # Display the frame
            cv2.imshow("Feed", disp_frame)

            # Press 'q' to Quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.waitKey(1) & 0xFF == ord('i') or cv2.waitKey(1) & 0xFF == ord('I'):
                cv2.imwrite("image.jpg", disp_frame)
                cap.release()

        # Release the Video Capture Object
        cap.release()

        # Destroy all cv2 Windows
        cv2.destroyAllWindows()


#  if __name__ == "__faceChecking__":
#      sys.exit(faceChecking() or 0)

faceCheckings()
