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

def generate_image(
    input_image,
    prompt,
    negative_prompt,
    steps,
    cfg,
    denoise,
    instantid_weight,
    seed,
    sampler_name,
    scheduler
):   
    with open("base_workflow.json", "r") as file_json:
        workflow = json.load(file_json)
    
    # Update KSampler parameters
    workflow["62"]["inputs"].update({
        "seed": seed if seed != -1 else random.randint(1, 1500000),
        "steps": steps,
        "cfg": cfg,
        "denoise": denoise,
        "sampler_name": sampler_name,
        "scheduler": scheduler
    })
    
    # Update prompt and negative prompt
    workflow["71"]["inputs"]["text"] = prompt
    workflow["72"]["inputs"]["text"] = negative_prompt
    
    # Update InstantID weight
    workflow["73"]["inputs"]["weight"] = instantid_weight
    
    global cached_seed
    if cached_seed == workflow["62"]["inputs"]["seed"]:
        return get_latest_image(OUTPUT_DIR)
    cached_seed = workflow["62"]["inputs"]["seed"]
    
    image = Image.fromarray(input_image)
    min_side = min(image.size)
    scale_factor = 512 / min_side
    new_size = (round(image.size[0] * scale_factor), round(image.size[1] * scale_factor))
    resized_image = image.resize(new_size)
    
    resized_image.save(os.path.join(INPUT_DIR, "test_api.jpg"))
    
    previous_image = get_latest_image(OUTPUT_DIR)
    start_queue(workflow)
    
    while True:
        latest_image = get_latest_image(OUTPUT_DIR)
        if latest_image != previous_image:
            current_working_directory = os.getcwd()
            new_image_path = os.path.join(current_working_directory, os.path.basename(latest_image))
            shutil.copy(latest_image, new_image_path)
            return new_image_path
        time.sleep(1)

# Define available samplers and schedulers
SAMPLERS = ["dpmpp_2s_ancestral", "euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral"]
SCHEDULERS = ["karras", "normal", "simple", "ddim_uniform"]

# Create the Gradio interface with additional parameters
img2img_demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Image(label="Input Image"),
        gr.Textbox(
            label="Prompt",
            value="korean, wearing suit and facing forward, ((over the shoulder shot)), plain background, eye contact"
        ),
        gr.Textbox(
            label="Negative Prompt",
            value="stock image, stock photo, text, sexual, magazine"
        ),
        gr.Slider(
            label="Steps",
            minimum=1,
            maximum=100,
            step=1,
            value=25
        ),
        gr.Slider(
            label="CFG Scale",
            minimum=1,
            maximum=20,
            step=0.5,
            value=5
        ),
        gr.Slider(
            label="Denoise Strength",
            minimum=0,
            maximum=1,
            step=0.05,
            value=0.9
        ),
        gr.Slider(
            label="InstantID Weight",
            minimum=0,
            maximum=1,
            step=0.05,
            value=0.8
        ),
        gr.Number(
            label="Seed (-1 for random)",
            value=-1
        ),
        gr.Dropdown(
            label="Sampler",
            choices=SAMPLERS,
            value="dpmpp_2s_ancestral"
        ),
        gr.Dropdown(
            label="Scheduler",
            choices=SCHEDULERS,
            value="karras"
        )
    ],
    outputs=["image"],
    title="영정사진 서비스",
    description="사용자가 원하는대로 params조절해서 이미지 생성이 가능합니다."
)




# demo.launch(allowed_paths=["/home/eardream2/ComfyUI/output"], share=True)
