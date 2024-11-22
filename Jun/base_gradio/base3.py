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

# 기본 프롬프트 설정
BASE_PROMPT = "professional portrait, studio lighting, formal suit, neutral background, confident expression, clean and polished look, well-groomed hair, half body, front view, focus on face, slight exposure correction, sharp focus, highly detailed, 4k, high resolution, center"
BASE_NEGATIVE = "ac_neg1, pointed chin, nevus, beard, naked, big ears, nude, naked, exposed body, bare skin, revealing clothes, suggestive, explicit, stain, ink, trouble, flip out, baby hair, flyaway, cross-eyed, strabismus"

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

def combine_prompts(additional_prompt):
    if additional_prompt.strip():
        return f"{BASE_PROMPT}, {additional_prompt}"
    return BASE_PROMPT

def generate_image(
    input_image,
    gender_option,
    age_option,
    additional_prompt,
    additional_negative,
    resolution,
    steps,
    denoise,
    instantid_weight,
    instantid_start_at,
    instantid_end_at,
    pulid_weight,
    pulid_start_at,
    pulid_end_at,
    face_mask_blur,
    face_mask_padding,
    face_mask_expand,
    seed
):   
    with open("1112_PuLID_InstantID_SWAP_workflow _api.json", "r") as file_json:
        workflow = json.load(file_json)
    
    # output 해상도 설정
    width, height = map(int, resolution.split('x'))
    workflow["45"]["inputs"].update({
        "width": width,
        "height": height
    })

    # output 해상도 설정
    width, height = map(int, resolution.split('x'))
    workflow["45"]["inputs"].update({
        "width": width,
        "height": height
    })
    workflow["83"]["inputs"].update({
        "side_length": max(width, height)  # 83번 노드는 longest side를 기준으로 스케일링
    })
    
    # Update KSampler parameters
    workflow["49"]["inputs"].update({
        "seed": seed if seed != -1 else random.randint(1, 1500000),
        "steps": steps,
        "denoise": denoise,
    })
    
    # 연령대별 프롬프트 매핑
    age_prompts = {
        "유아": "young child, baby face, innocent look, soft features, 3-6 years old",
        "청소년": "teenager, young face, youthful features, 13-19 years old",
        "중년": "middle aged, mature features, professional look, 40-55 years old",
        "노년": "elderly, senior, aged features, dignified look, over 60 years old"
    }
    
    # Update prompts
    # 42번 노드의 기존 프롬프트는 유지하고 나이와 추가 프롬프트를 덧붙임
    base_prompt = workflow["42"]["inputs"]["text"]
    age_prompt = age_prompts[age_option]
    final_prompt = f"{base_prompt}, {age_prompt}"
    if additional_prompt.strip():
        final_prompt = f"{final_prompt}, {additional_prompt}"
    workflow["42"]["inputs"]["text"] = final_prompt
    
    # Update gender in node 26
    workflow["26"]["inputs"]["preset_expr"] = "#Female > #Male" if gender_option == "woman" else "#Female < #Male"
    
    # Update negative prompts
    negative_prompt = f"{BASE_NEGATIVE}, {additional_negative}" if additional_negative.strip() else BASE_NEGATIVE
    workflow["2"]["inputs"]["text"] = negative_prompt
    workflow["41"]["inputs"]["text"] = negative_prompt
    
    # Update InstantID parameters
    workflow["58"]["inputs"].update({
        "weight": instantid_weight,
        "start_at": instantid_start_at,
        "end_at": instantid_end_at
    })
    
    # Update PuLID parameters
    workflow["75"]["inputs"].update({
        "weight": pulid_weight,
        "start_at": pulid_start_at,
        "end_at": pulid_end_at
    })
    
    # Update face mask parameters
    workflow["59"]["inputs"]["amount"] = face_mask_blur
    workflow["43"]["inputs"]["padding"] = face_mask_padding
    workflow["63"]["inputs"]["expand"] = face_mask_expand
    
    global cached_seed
    if cached_seed == workflow["49"]["inputs"]["seed"]:
        return get_latest_image(OUTPUT_DIR)
    cached_seed = workflow["49"]["inputs"]["seed"]
    
    image = Image.fromarray(input_image)
    min_side = min(image.size)
    scale_factor = 1024 / min_side
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

# Define available options
RESOLUTIONS = ["1024x1024", "1216x832", "832x1216"]  # Square, Landscape, Portrait

# Create the Gradio interface
with gr.Blocks(title="ID Photo Generator") as demo:
    gr.Markdown("# ID Photo Generator")
    gr.Markdown("Generate professional ID photos with customizable parameters.")
    
    with gr.Row():
        # Left Column - Input
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input Image")
            
            with gr.Row():
                gender = gr.Radio(
                    label="Gender",
                    choices=["man", "woman"],
                    value="man",
                    type="value"
                )
                age = gr.Radio(
                    label="Age Group",
                    choices=["유아", "청소년", "중년", "노년"],
                    value="중년",
                    type="value"
                )
            
            additional_prompt = gr.Textbox(
                label="Additional Prompt (Optional)",
                placeholder="Add your custom prompt here..."
            )
            additional_negative = gr.Textbox(
                label="Additional Negative Prompt (Optional)",
                placeholder="Add your custom negative prompt here..."
            )
            
            resolution = gr.Radio(
                label="Output Resolution",
                choices=RESOLUTIONS,
                value="1024x1024",
                type="value"
            )
            
            with gr.Row():
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, step=1, value=10)
            
            with gr.Row():
                denoise = gr.Slider(label="Denoise Strength", minimum=0.8, maximum=1, step=0.01, value=0.9)
                seed = gr.Number(label="Seed (-1 for random)", value=-1)
        
        # Right Column - Output and Advanced Settings
        with gr.Column(scale=1):
            output_image = gr.Image(label="Output Image")
            
            with gr.Accordion("Advanced Settings", open=False):
                gr.Markdown("### ID Model Controls")
                instantid_weight = gr.Slider(label="InstantID Weight", minimum=0, maximum=2, step=0.1, value=1.2)
                with gr.Row():
                    instantid_start_at = gr.Slider(label="Start", minimum=0, maximum=1, step=0.1, value=0)
                    instantid_end_at = gr.Slider(label="End", minimum=0, maximum=1, step=0.1, value=1)
                
                gr.Markdown("### PuLID Controls")
                pulid_weight = gr.Slider(label="PuLID Weight", minimum=0, maximum=2, step=0.1, value=1)
                with gr.Row():
                    pulid_start_at = gr.Slider(label="Start", minimum=0, maximum=1, step=0.1, value=0)
                    pulid_end_at = gr.Slider(label="End", minimum=0, maximum=1, step=0.1, value=0.9)
                
                gr.Markdown("### Face Mask Controls")
                face_mask_blur = gr.Slider(label="Face Mask Blur", minimum=0, maximum=128, step=1, value=64)
                with gr.Row():
                    face_mask_padding = gr.Slider(label="Padding", minimum=0, maximum=300, step=10, value=50)
                    face_mask_expand = gr.Slider(label="Expand", minimum=0, maximum=100, step=1, value=30)
    
    generate_btn = gr.Button("Generate", variant="primary", size="lg")
    generate_btn.click(
        fn=generate_image,
        inputs=[
            input_image, gender, age, additional_prompt, additional_negative, resolution,
            steps, denoise,
            instantid_weight, instantid_start_at, instantid_end_at,
            pulid_weight, pulid_start_at, pulid_end_at,
            face_mask_blur, face_mask_padding, face_mask_expand,
            seed,
        ],
        outputs=output_image
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True)