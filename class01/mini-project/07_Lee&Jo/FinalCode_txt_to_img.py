import os
import time
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import torch
import speech_recognition as sr
import ipywidgets as widgets
from my_module import create_ov_pipe
from IPython.display import display

# Define load_image function using OpenCV
def load_image(uri):
    """
    Load an image from the given URI.

    Args:
        uri (str): The URI of the image.

    Returns:
        numpy.ndarray: The loaded image.
    """
    # Read the image using OpenCV
    image = cv2.imread(uri)
    if image is None:
        raise FileNotFoundError(f"Failed to load image from {uri}")
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Define U2NET and U2NETP classes here or import from local path
from model.u2net import U2NET, U2NETP

model_config = namedtuple("ModelConfig", ["name", "url", "model", "model_args"])

u2net_lite = model_config(
    name="u2net_lite",
    url="https://drive.google.com/uc?id=1W8E4FHIlTVstfRkYmNOjbr0VDXTZm0jD",
    model=U2NETP,
    model_args=(),
)
u2net = model_config(
    name="u2net",
    url="https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
    model=U2NET,
    model_args=(3, 1),
)
u2net_human_seg = model_config(
    name="u2net_human_seg",
    url="https://drive.google.com/uc?id=1m_Kgs91b21gayc2XLW0ou8yugAIadWVP",
    model=U2NET,
    model_args=(3, 1),
)

# Set u2net_model to one of the three configurations listed above.
u2net_model = u2net_lite

# The filenames of the downloaded and converted models.
MODEL_DIR = "model"
model_path = Path(MODEL_DIR) / u2net_model.name / Path(u2net_model.name).with_suffix(".pth")

if not model_path.exists():
    import gdown

    os.makedirs(name=model_path.parent, exist_ok=True)
    print("Start downloading model weights file... ")
    with open(model_path, "wb") as model_file:
        gdown.download(url=u2net_model.url, output=model_file)
        print(f"Model weights have been downloaded to {model_path}")

# Load the model.
net = u2net_model.model(*u2net_model.model_args)
net.eval()

# Load the weights.
print(f"Loading model weights from: '{model_path}'")
net.load_state_dict(state_dict=torch.load(model_path, map_location="cpu"))

model_ir = ov.convert_model(net, example_input=torch.zeros((1,3,512,512)), input=([1, 3, 512, 512]))

def recognize_speech(recognizer):
    with sr.Microphone() as source:
        print("Speak something...")
        audio = recognizer.listen(source)

    try:
        # Google Speech Recognition API를 사용하여 음성을 텍스트로 변환
        text = recognizer.recognize_google(audio, language='en-US')
        print("Speech Recognition Result:", text)
        return text
    except sr.UnknownValueError:
        print("Could not understand the audio.")
        return ""
    except sr.RequestError as e:
        print(f"Could not access Google Speech Recognition API. Error: {e}")
        return ""

def save_to_file(text, filename="output.txt"):
    with open(filename, "w") as file:  
        file.write(text + "\n")  
    print(f"Text has been saved to {filename} file.")

recognizer = sr.Recognizer()
text = recognize_speech(recognizer)
if text:
    save_to_file(text)

# output.txt 파일을 읽어와서 변수에 저장
with open('output.txt', 'r') as file:
    sample_text = file.read()

# 위에서 읽은 내용을 text_prompt 변수의 초기값으로 설정
text_prompt = widgets.Text(value=sample_text, description='your text')
num_steps = widgets.IntSlider(min=1, max=50, value=10, description='steps:')
seed = widgets.IntSlider(min=0, max=10000000, description='seed: ', value=10)
display(widgets.VBox([text_prompt, seed, num_steps]))

ov_pipe = create_ov_pipe()
result = ov_pipe(text_prompt.value, num_inference_steps=num_steps.value, seed=seed.value)

final_image = result['sample'][0]
if result['iterations']:
    all_frames = result['iterations']
    img = next(iter(all_frames))
    img.save(fp='result.gif', format='GIF', append_images=iter(all_frames), save_all=True, duration=len(all_frames) * 5, loop=0)
final_image.save('final.png')

# Display the final image
final_image.show()

cap = cv2.VideoCapture(0)

input_mean = np.array([123.675, 116.28 , 103.53]).reshape(1, 3, 1, 1)
input_scale = np.array([58.395, 57.12 , 57.375]).reshape(1, 3, 1, 1)

# Load background image
BACKGROUND_FILE = "final.png"
background_image = cv2.imread(BACKGROUND_FILE)
background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
background_image = cv2.resize(background_image, (640, 480))  # Resize to match webcam image size

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image shape to a shape and a data type expected by the network
    # for OpenVINO IR model: (1, 3, 512, 512).
    resized_image = cv2.resize(src=frame, dsize=(512, 512))
    input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)
    input_image = (input_image - input_mean) / input_scale

    core = ov.Core()
    # Load the network to OpenVINO Runtime.
    compiled_model_ir = core.compile_model(model=model_ir, device_name='CPU')
    # Get the names of input and output layers.
    input_layer_ir = compiled_model_ir.input(0)
    output_layer_ir = compiled_model_ir.output(0)

    # Do inference on the input image.
    result = compiled_model_ir([input_image])[output_layer_ir]

    # Resize the network result to the image shape and round the values
    # to 0 (background) and 1 (foreground).
    # The network result has (1,1,512,512) shape. The `np.squeeze` function converts this to (512, 512).
    resized_result = np.rint(
        cv2.resize(src=np.squeeze(result), dsize=(frame.shape[1], frame.shape[0]))
    ).astype(np.uint8)

    # Create a copy of the image and set all background values to 255 (white).
    bg_removed_result = frame.copy()
    bg_removed_result[resized_result == 0] = 255

    # Overlay the foreground on the background image
    new_image = background_image.copy()
    new_image[resized_result == 1] = bg_removed_result[resized_result == 1]

    # Display the original image and the image with the new background side by side
    cv2.imshow('Original', frame)
    cv2.imshow('Background Removed', new_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()