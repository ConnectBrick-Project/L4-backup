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

# ComfyUI 경로
COMFY_OUTPUT_DIR = "/home/eardream2/ComfyUI/output"
COMFY_INPUT_DIR = "/home/eardream2/ComfyUI/input"

# 결과물 저장할 경로
SAVE_DIR = "/home/eardream2/Jun/output"  # 원하는 저장 경로로 변경하세요

def ensure_directory(directory):
    """디렉토리가 없으면 생성"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

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
        ensure_directory(SAVE_DIR)
        
        # 워크플로우 로드 및 설정
        with open("1112_PuLID_InstantID_swap_workflow.json", "r") as file_json:
            workflow = json.load(file_json)
        
        # 파라미터 설정
        actual_seed = seed if seed != -1 else random.randint(1, 1500000)
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
        
        # 입력 이미지 처리
        image = Image.fromarray(input_image)
        min_side = min(image.size)
        scale_factor = 1024 / min_side
        new_size = (round(image.size[0] * scale_factor), round(image.size[1] * scale_factor))
        resized_image = image.resize(new_size)
        
        # ComfyUI input에 저장
        comfy_input_path = os.path.join(COMFY_INPUT_DIR, "test_api.jpg")
        resized_image.save(comfy_input_path)
        
        # 이미지 생성 시작
        start_time = time.time()
        start_queue(workflow)
        
        # 결과 대기
        comfy_output_path = wait_for_new_image(start_time)
        
        if comfy_output_path and os.path.exists(comfy_output_path):
            # 결과물을 원하는 경로에 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f"result_{timestamp}.png"
            save_path = os.path.join(SAVE_DIR, save_filename)
            shutil.copy(comfy_output_path, save_path)
            
            # Gradio에 경로 반환
            return save_path
            
        raise Exception("Generated image not found")
        
    except Exception as e:
        print(f"Error in generate_image: {str(e)}")
        raise gr.Error(f"Image generation failed: {str(e)}")

def wait_for_new_image(start_time):
    """ComfyUI output 디렉토리에서 새 이미지 대기"""
    timeout = 300  # 5분 타임아웃
    while True:
        if time.time() - start_time > timeout:
            raise Exception("Image generation timed out")
        
        image_files = []
        for root, dirs, files in os.walk(COMFY_OUTPUT_DIR):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    if os.path.getctime(full_path) > start_time:
                        image_files.append(full_path)
        
        if image_files:
            return max(image_files, key=os.path.getctime)
        
        time.sleep(1)

def start_queue(prompt_workflow):
    """ComfyUI API에 작업 요청"""
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    requests.post("http://localhost:5001/api/prompt", data=data)

# Gradio 인터페이스 설정
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
        gr.Slider(label="Steps", minimum=1, maximum=100, step=1, value=10),
        gr.Slider(label="CFG Scale", minimum=1, maximum=20, step=0.5, value=1),
        gr.Slider(label="Denoise Strength", minimum=0, maximum=1, step=0.05, value=0.5),
        gr.Slider(label="InstantID Weight", minimum=0, maximum=2, step=0.1, value=1.2),
        gr.Number(label="Seed (-1 for random)", value=-1),
        gr.Dropdown(
            label="Sampler",
            choices=["euler_ancestral", "euler", "heun", "dpm_2", "dpm_2_ancestral", "dpmpp_sde", "dpmpp_2s_ancestral"],
            value="euler_ancestral"
        ),
        gr.Dropdown(
            label="Scheduler",
            choices=["karras", "normal", "simple", "sgm_uniform"],
            value="sgm_uniform"
        )
    ],
    outputs=["image"],
    title="ID Photo Generator",
    description="Generate professional ID photos with customizable parameters."
)

if __name__ == "__main__":
    demo.launch(share=True, allowed_paths=[SAVE_DIR])