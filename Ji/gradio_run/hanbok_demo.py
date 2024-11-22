import json
import os
import time
import random
import gradio as gr
import requests
from PIL import Image


URL = "http://localhost:8188/api/prompt" # ComfyUI 실행 서버 주소에 따라 수정 필요
OUTPUT_DIR = "/project-root/ComfyUI/output" # ComfyUI 가 설치되어있는 서버에 따라 수정 필요
INPUT_DIR = "project-root/ComfyUI/input" # ComfyUI 가 설치되어있는 서버에 따라 수정 필요
SAVED_IMAGES_DIR = "/project-root/saved_outputs"

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

def generate_image(
    input_image,
    gender,
    age,
    color,
    filename,
    steps,
    cfg,
    denoise,
    seed
):   
    with open("hanbok_change_api_final.json", "r") as file_json:
        workflow = json.load(file_json)
    
    # Update KSampler parameters
    workflow["37"]["inputs"].update({
        "seed": seed if seed != -1 else random.randint(1, 1500000),
        "steps": steps,
        "cfg": cfg,
        "denoise": denoise
    })
    
    # Update prompt
    workflow["33"]["inputs"]["text"] = f"A portrait of a {age} years old Korean {gender} wearing {color} hanbok, over the shoulder, white background"
    
    global cached_seed
    if cached_seed == workflow["37"]["inputs"]["seed"]:
        latest_image_path = get_latest_image(OUTPUT_DIR)
        latest_image = Image.open(latest_image_path)
        return latest_image
    cached_seed = workflow["37"]["inputs"]["seed"]
    
    # Save the input image
    image = Image.fromarray(input_image)
    image.save(os.path.join(INPUT_DIR, "cloth_change.png"))
    
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
hanbok_demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Image(label="Input Image"),
        gr.Radio(label="Gender", choices=['man', 'woman', 'boy', 'girl'], value="man"),
        gr.Textbox(label="Age", value="60"),
        gr.Textbox(label="Hanbok color", value="blue"),
        gr.Textbox(label='File name', value='hanbok_change_demo'),
        gr.Slider(label="Steps", minimum=10, maximum=50, step=1, value=20),
        gr.Slider(label="CFG Scale", minimum=3, maximum=9, step=0.5, value=4),
        gr.Slider(label="Denoise Strength", minimum=0.7, maximum=1, step=0.05, value=0.8),
        gr.Number(label="Seed (-1 for random)", value=-1)
    ],
    outputs=[gr.Image(label="Generated Image")],
    title="한복 변경",
    description="대상자의 성별, 나이, 원하는 한복의 색을 입력하면 한복 적용이 가능합니다."
)