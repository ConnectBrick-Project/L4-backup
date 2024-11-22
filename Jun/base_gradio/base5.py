import json
import os
import time
import random
import gradio as gr
import numpy as np
import requests
from PIL import Image
import shutil
from datetime import datetime
import glob

URL = "http://localhost:5001/api/prompt"
OUTPUT_DIR = "/home/eardream2/ComfyUI/output"
INPUT_DIR = "/home/eardream2/ComfyUI/input"

def clean_output_directory():
    """Clean up the output directory before new generation"""
    for file in glob.glob(os.path.join(OUTPUT_DIR, "*")):
        try:
            os.remove(file)
        except Exception as e:
            print(f"Error cleaning file {file}: {e}")

def wait_for_new_image(start_time):
    """Wait for a new image to appear in the output directory"""
    timeout = 300  # 5 minutes timeout
    while True:
        if time.time() - start_time > timeout:
            raise Exception("Image generation timed out")
        
        files = glob.glob(os.path.join(OUTPUT_DIR, "*.png"))
        if files:
            # Sort by creation time and get the newest
            newest_file = max(files, key=os.path.getctime)
            if os.path.getctime(newest_file) > start_time:
                return newest_file
        time.sleep(1)

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
    try:
        # Clean output directory first
        clean_output_directory()
        
        # Load workflow
        with open("1112_PuLID_InstantID_swap_workflow.json", "r") as file_json:
            workflow = json.load(file_json)
        
        # Generate random seed if needed
        actual_seed = seed if seed != -1 else random.randint(1, 1500000)
        
        # Update workflow parameters
        workflow["49"]["inputs"].update({
            "seed": actual_seed,
            "steps": steps,
            "cfg": cfg,
            "denoise": denoise,
            "sampler_name": sampler_name,
            "scheduler": scheduler
        })
        
        workflow["42"]["inputs"]["text"] = prompt
        workflow["41"]["inputs"]["text"] = negative_prompt
        workflow["58"]["inputs"]["weight"] = instantid_weight
        
        # Process input image
        image = Image.fromarray(input_image)
        min_side = min(image.size)
        scale_factor = 1024 / min_side
        new_size = (round(image.size[0] * scale_factor), round(image.size[1] * scale_factor))
        resized_image = image.resize(new_size)
        
        # Save input image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = os.path.join(INPUT_DIR, "test_api.jpg")
        resized_image.save(input_path)
        
        # Record start time and queue the job
        start_time = time.time()
        start_queue(workflow)
        
        # Wait for the new image
        generated_image_path = wait_for_new_image(start_time)
        
        # Copy to current directory with unique name
        output_filename = f"result_{timestamp}.png"
        output_path = os.path.join(os.getcwd(), output_filename)
        shutil.copy(generated_image_path, output_path)
        
        return output_path
        
    except Exception as e:
        print(f"Error in generate_image: {str(e)}")
        raise gr.Error(f"Image generation failed: {str(e)}")

def start_queue(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    requests.post(URL, data=data)

# Define available samplers and schedulers
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

if __name__ == "__main__":
    demo.launch(share=True)