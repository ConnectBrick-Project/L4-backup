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
    with open("1112_PuLID_InstantID_swap_workflow.json", "r") as file_json:
        workflow = json.load(file_json)
    
    # Update KSampler parameters
    workflow["49"]["inputs"].update({
        "seed": seed if seed != -1 else random.randint(1, 1500000),
        "steps": steps,
        "cfg": cfg,
        "denoise": denoise,
        "sampler_name": sampler_name,
        "scheduler": scheduler
    })
    
    # Update prompt and negative prompt
    workflow["42"]["inputs"]["text"] = prompt
    workflow["41"]["inputs"]["text"] = negative_prompt
    
    # Update InstantID weight
    workflow["58"]["inputs"]["weight"] = instantid_weight
    
    global cached_seed
    if cached_seed == workflow["49"]["inputs"]["seed"]:
        return get_latest_image(OUTPUT_DIR)
    cached_seed = workflow["49"]["inputs"]["seed"]
    
    image = Image.fromarray(input_image)
    min_side = min(image.size)
    scale_factor = 1024 / min_side  # Changed to 1024 as per workflow
    new_size = (round(image.size[0] * scale_factor), round(image.size[1] * scale_factor))
    resized_image = image.resize(new_size)
    
    resized_image.save(os.path.join(INPUT_DIR, "test_api.jpg"))  # Changed filename as per workflow
    
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

# Define available samplers and schedulers based on workflow
SAMPLERS = ["euler_ancestral", "euler", "heun", "dpm_2", "dpm_2_ancestral", "dpmpp_sde", "dpmpp_2s_ancestral"]
SCHEDULERS = ["karras", "normal", "simple", "sgm_uniform"]

# Create the Gradio interface
demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Image(label="Input Image"),
        gr.Textbox(
            label="Prompt",
            value="professional portrait, studio lighting, formal suit, neutral background, confident expression"
        ),
        gr.Textbox(
            label="Negative Prompt",
            value="ac_neg1, pointed chin, nevus, beard, naked, big ears, nude, naked, exposed body, bare skin, revealing clothes, suggestive, explicit"
        ),
        gr.Slider(
            label="Steps",
            minimum=1,
            maximum=100,
            step=1,
            value=10
        ),
        gr.Slider(
            label="CFG Scale",
            minimum=1,
            maximum=20,
            step=0.5,
            value=1
        ),
        gr.Slider(
            label="Denoise Strength",
            minimum=0,
            maximum=1,
            step=0.05,
            value=0.5
        ),
        gr.Slider(
            label="InstantID Weight",
            minimum=0,
            maximum=2,
            step=0.1,
            value=1.2
        ),
        gr.Number(
            label="Seed (-1 for random)",
            value=-1
        ),
        gr.Dropdown(
            label="Sampler",
            choices=SAMPLERS,
            value="euler_ancestral"
        ),
        gr.Dropdown(
            label="Scheduler",
            choices=SCHEDULERS,
            value="sgm_uniform"
        )
    ],
    outputs=["image"],
    title="ID Photo Generator",
    description="Generate professional ID photos with customizable parameters."
)

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True)
