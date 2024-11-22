import json
import os
import time
import random
import gradio as gr
import numpy as np
import requests
from PIL import Image
import shutil

URL = "http://localhost:8188/api/prompt"
OUTPUT_DIR = "/home/eardream2/ComfyUI/output"
INPUT_DIR = "/home/eardream2/ComfyUI/input"

def get_latest_image(folder):
    files = os.listdir(folder)
    image_files = [f for f in files if f.lower().endswith(('.png','.jpg','.jpeg','.webp'))]
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    return os.path.join(folder, image_files[-1]) if image_files else None

def start_queue(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    requests.post(URL, data=data)

def generate_image(
    input_image,
    smile_strength,
    eye_open,
    eye_size,
    brow_raise,
    brow_lower,
    mouth_open,
    mouth_size,
    filename
):   
    with open("/home/eardream2/su/workflow/face_api.json", "r") as file:
        workflow = json.load(file)
    
    # Update ExpressionEditor parameters
    workflow["6"]["inputs"].update({
        "smile": min(smile_strength, 1.3),  # 최대값을 1.3으로 제한
        "eye_open": min(eye_open, 1.3),  # 다른 파라미터들도 1.3으로 제한
        "eye_size": max(-1.3, min(eye_size, 1.3)),
        "brow_raise": min(brow_raise, 1.3),
        "brow_lower": min(brow_lower, 1.3),
        "mouth_open": min(mouth_open, 1.3),
        "mouth_size": max(-1.3, min(mouth_size, 1.3))
    })
    
    image = Image.fromarray(input_image)
    image.save(os.path.join(INPUT_DIR, "input_image.png"))
    
    # Update LoadImage node with the input image
    workflow["3"]["inputs"]["image"] = "input_image.png"
    
    previous_image = get_latest_image(OUTPUT_DIR)
    start_queue(workflow)
    
    while True:
        latest_image = get_latest_image(OUTPUT_DIR)
        if latest_image != previous_image:
            new_image_path = os.path.join(os.getcwd(), f'{filename}.png')
            shutil.copy(latest_image, new_image_path)
            return new_image_path, new_image_path
        time.sleep(1)

face = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Image(label="Input Image"),
        gr.Slider(label="Smile Strength", minimum=0, maximum=1.3, step=0.1, value=1.0),
        gr.Slider(label="Eye Open", minimum=0, maximum=1.3, step=0.1, value=0.5),
        gr.Slider(label="Eye Size", minimum=-1.3, maximum=1.3, step=0.1, value=0),
        gr.Slider(label="Brow Raise", minimum=0, maximum=1.3, step=0.1, value=0.0),
        gr.Slider(label="Brow Lower", minimum=0, maximum=1.3, step=0.1, value=0.0),
        gr.Slider(label="Mouth Open", minimum=0, maximum=1.3, step=0.1, value=0),
        gr.Slider(label="Mouth Size", minimum=-1.3, maximum=1.3, step=0.1, value=0),
        gr.Textbox(label='File name', value='expression_edited_image')
    ],
    outputs=[gr.Image(label="Generated Image"), gr.Textbox(label="Saved Path")],
    title="Expression Editor",
    description="입력 이미지의 표정을 조정합니다. 각 파라미터의 최대값은 1.3입니다."
)

face.launch()