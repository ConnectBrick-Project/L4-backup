import json
import os
import time
import random
import gradio as gr
import numpy as np
import requests
from PIL import Image
import random
from datetime import datetime


URL = "http://localhost:8188/api/prompt" # ComfyUI 실행 서버 주소에 따라 수정 필요
OUTPUT_DIR = "/project-root/ComfyUI/output" # ComfyUI 가 설치되어있는 서버에 따라 수정 필요
INPUT_DIR = "project-root/ComfyUI/input" # ComfyUI 가 설치되어있는 서버에 따라 수정 필요
SAVED_IMAGES_DIR = "/project-root/saved_outputs"
HANBOK_DIR = "/project-root/hanbok_stock"

cached_seed = 0

def get_latest_image(folder):
    files = os.listdir(folder)
    image_files = [f for f in files if f.lower().endswith(('.png','.jpg','.jpeg','.webp'))]
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    latest_image = os.path.join(folder, image_files[-1]) if image_files else None
    return latest_image

def start_queue(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    requests.post(URL, data=data)

def select_hanbok(gender, color):
    files_gender = os.listdir(HANBOK_DIR + f"/{gender}")
    color_f = color.lower()
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    
    hanbok_files = [
        f for f in files_gender 
        if f.lower().startswith(color_f) and f.lower().endswith(valid_extensions)
    ]
    if not hanbok_files:
        return None  # Return None or raise an exception if no matching files are found
    hanbok_image = random.choice(hanbok_files)
    hanbok_path = HANBOK_DIR + f"/{gender}/{hanbok_image}"
    return hanbok_path


def generate_image(
    input_image,
    gender,
    color,
    filename,
    seed
):   
    with open("hanbok_stock_api.json", "r") as file_json:
        workflow = json.load(file_json)
    
    # Update KSampler parameters
    workflow["88"]["inputs"].update({
        "seed": seed if seed != -1 else random.randint(1, 1500000),
    })
    
    global cached_seed
    if cached_seed == workflow["88"]["inputs"]["seed"]:
        latest_image_path = get_latest_image(OUTPUT_DIR)
        latest_image = Image.open(latest_image_path)
        return latest_image
    cached_seed = workflow["88"]["inputs"]["seed"]
    
    # Save the input image
    image = Image.fromarray(input_image)
    image.save(os.path.join(INPUT_DIR, "cloth_change_stock.png"))

    hanbok_image_path = select_hanbok(gender, color)
    selected_hanbok_image = Image.open(hanbok_image_path)
    selected_hanbok_image.save(os.path.join(INPUT_DIR, "selected_hanbok_image.png"))
    
    previous_image = get_latest_image(OUTPUT_DIR)
    start_queue(workflow)
    
    while True:
        latest_image_path = get_latest_image(OUTPUT_DIR)
        if latest_image_path != previous_image:
            # Add delay logic here to ensure the file is fully written
            previous_size = -1
            while True:
                current_size = os.path.getsize(latest_image_path)
                if current_size == previous_size:
                    break  # File size has stabilized
                previous_size = current_size
                time.sleep(0.5)
            
            # Now open the image safely
            with open(latest_image_path, "rb") as f:
                latest_image = Image.open(f)
                latest_image.load()  # Ensure the image is fully loaded
            
            # Save the image with the specified filename
            new_image_path = os.path.join(SAVED_IMAGES_DIR, f'{filename}.png')
            latest_image.save(new_image_path)
            return new_image_path
        time.sleep(1)


# Create the Gradio interface with additional parameters
hanbok_stock = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Image(label="인물 사진"),
        gr.Radio(label="Gender", choices=['man', 'woman', 'boy', 'girl'], value="man"),
        gr.Radio(label="Hanbok color", choices=['Blue','Pink','Green','White','Purple'], value="Blue"),
        gr.Textbox(label='File name', value='hanbok_change_stock'),
        gr.Number(label="Seed (-1 for random)", value=-1)
    ],
    outputs=[gr.Image(label="Generated Image")],
    title="한복 변경",
    description="대상자의 성별, 나이, 원하는 한복의 색을 입력하면 한복 적용이 가능합니다."
)