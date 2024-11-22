import json
import os
import time
import random


import gradio as gr
import numpy as np
import requests

from PIL import Image
import shutil


URL = "http://localhost:5001/api/prompt"
OUTPUT_DIR = "/home/eardream2/ComfyUI/output"
INPUT_DIR = "/home/eardream2/ComfyUI/input"



cached_seed = 0

def get_latest_image(folder):
    files = os.listdir(folder)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    latest_image = os.path.join(folder, image_files[-1]) if image_files else None
    return latest_image


def start_queue(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    requests.post(URL, data=data)


def generate_image(input_image):   
    with open("base_workflow.json", "r") as file_json:
        prompt = json.load(file_json)

    prompt["62"]["inputs"]["seed"] = random.randint(1, 1500000)
    global cached_seed
    if cached_seed == prompt["62"]["inputs"]["seed"]:
        return get_latest_image(OUTPUT_DIR)
    cached_seed = prompt["62"]["inputs"]["seed"]
    
    image = Image.fromarray(input_image)
    min_side = min(image.size)
    scale_factor = 512 / min_side
    new_size = (round(image.size[0] * scale_factor), round(image.size[1] * scale_factor))
    resized_image = image.resize(new_size)

    resized_image.save(os.path.join(INPUT_DIR, "test_api.jpg"))

    previous_image = get_latest_image(OUTPUT_DIR)
    
    start_queue(prompt)

    # 새로운 이미지를 생성할 때까지 기다리는 루프
    while True:
        latest_image = get_latest_image(OUTPUT_DIR)
        if latest_image != previous_image:
            # 새로운 이미지가 발견되면 현재 작업 디렉토리로 복사
            current_working_directory = os.getcwd()
            new_image_path = os.path.join(current_working_directory, os.path.basename(latest_image))
            shutil.copy(latest_image, new_image_path)
            
            # 복사한 이미지 경로를 반환
            return new_image_path

        time.sleep(1)

demo = gr.Interface(fn=generate_image, inputs=["image"], outputs=["image"])
demo.launch(allowed_paths=["/home/eardream2/ComfyUI/output"],share=True)
