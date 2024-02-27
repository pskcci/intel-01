import cv2
import speech_recognition as sr
import subprocess
import gradio as gr
import test1
from test1 import create_ov_pipe

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

def save_to_file(text, filename="sample.txt"):
    with open(filename, "a") as file:  
        file.write(text + "\n")  
    print(f"Text has been saved to {filename} file.")

def display_image(image_path):
    subprocess.Popen(['xdg-open', image_path])

def display_webcam_with_text(text):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open the webcam.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read video frame.")
            break

        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def create_ov_pipe(text, num_inference_steps, seed):
    # 여기에 create_ov_pipe 함수의 내용을 작성해주세요.
    pass

def generate_from_text(text, seed, num_steps, _=gr.Progress(track_tqdm=True)):
    ov_pipe = create_ov_pipe
    result = ov_pipe(text, num_inference_steps=num_steps, seed=seed)
    return result["sample"][0]

if __name__ == "__main__":
    recognizer = sr.Recognizer()
    
    while True:
        user_input = input("Enter 'start' to begin speech recognition, or 'quit' to exit: ")

        if user_input == 'start':
            text = recognize_speech(recognizer)
            save_to_file(text)
            if text.lower() == "blue":
                display_image("blue.jpeg")
            elif text.lower() == "black":
                display_image("black.jpeg")
            elif text.lower() == "green":
                display_image("green.jpeg")
            elif text.lower() == "car":
                display_image("car.jpeg")
            elif text.lower() == "final":
                display_image("final.png")
            display_webcam_with_text(text)
        elif user_input == 'quit':
            print("Exiting the program.")
            break
        else:
            print("Please enter a valid command.")

    sample_text = open('sample.txt')
    text_data = sample_text.read()

    with gr.Blocks() as demo:
        with gr.Tab("Text-to-Image generation"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(lines=3, label="Text")
                    seed_input = gr.Slider(0, 10000000, value=42, label="Seed")
                    steps_input = gr.Slider(1, 50, value=20, step=1, label="Steps")
                out = gr.Image(label="Result", type="pil")
            btn = gr.Button()
            btn.click(generate_from_text, [text_input, seed_input, steps_input], out)
            gr.Examples([[text_data, 10, 10]], [text_input, seed_input, steps_input])

    try:
        demo.queue().launch(debug=True, inline=False)
    except Exception:
        demo.queue().launch(share=True, debug=True, inline=False)