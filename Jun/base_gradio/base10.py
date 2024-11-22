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

# ComfyUI 경로
URL = "http://localhost:5001/api/prompt"
COMFY_OUTPUT_DIR = "/home/eardream2/ComfyUI/output"
COMFY_INPUT_DIR = "/home/eardream2/ComfyUI/input"

# 저장할 경로
INPUT_SAVE_DIR = "/home/eardream2/Jun/input"   # 입력 이미지 저장 경로
OUTPUT_SAVE_DIR = "/home/eardream2/Jun/output" # 출력 이미지 저장 경로

# 기본 프롬프트 설정
BASE_PROMPT = "professional portrait, studio lighting, formal suit, neutral background, confident expression, clean and polished look, well-groomed hair, half body, front view, focus on face, slight exposure correction, sharp focus, highly detailed, 4k, high resolution, center"
BASE_NEGATIVE = "ac_neg1, pointed chin, nevus, beard, naked, big ears, nude, naked, exposed body, bare skin, revealing clothes, suggestive, explicit, stain, ink, trouble, flip out, baby hair, flyaway, cross-eyed, strabismus"

def ensure_directory(directory):
    """디렉토리가 없으면 생성"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def combine_prompts(base_prompt, additional_prompt):
    """기본 프롬프트와 추가 프롬프트를 결합"""
    if additional_prompt.strip():
        return f"{base_prompt}, {additional_prompt}"
    return base_prompt

def get_gender_prompt(gender_option):
    """성별에 따른 프롬프트 반환"""
    return "an woman" if gender_option == "woman" else "a man"

def generate_image(
    input_image,
    additional_prompt="",
    additional_negative="",
    steps=10,
    denoise=0.9,
    instantid_weight=1.2,
    seed=-1,
    seed_option="random",
    gender_option="man",
    age_option="중년",
    resolution="1024x1024",
    instantid_start_at=0,
    instantid_end_at=1,
    pulid_weight=1.0,
    pulid_start_at=0,
    pulid_end_at=0.9,
    face_mask_blur=64,
    face_mask_padding=50,
    face_mask_expand=30
):   
    try:
        # 저장 디렉토리 확인
        ensure_directory(INPUT_SAVE_DIR)
        ensure_directory(OUTPUT_SAVE_DIR)
        
        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 입력 이미지 처리 및 저장
        image = Image.fromarray(input_image)
        input_save_path = os.path.join(INPUT_SAVE_DIR, f"input_{timestamp}.png")
        image.save(input_save_path)
        print(f"Saved input image to: {input_save_path}")
        
        # ComfyUI 처리를 위한 리사이징
        min_side = min(image.size)
        scale_factor = 1024 / min_side
        new_size = (round(image.size[0] * scale_factor), round(image.size[1] * scale_factor))
        resized_image = image.resize(new_size)
        
        # ComfyUI input에 저장
        comfy_input_path = os.path.join(COMFY_INPUT_DIR, "test_api.jpg")
        resized_image.save(comfy_input_path)
        
        # 연령대별 프롬프트 매핑
        age_prompts = {
            "유아": "young child, baby face, innocent look, soft features, 3-6 years old",
            "청소년": "teenager, young face, youthful features, 13-19 years old",
            "중년": "middle aged, mature features, professional look, 40-55 years old",
            "노년": "elderly, senior, aged features, dignified look, over 60 years old"
        }
        
        # 워크플로우 설정
        with open("1112_PuLID_InstantID_SWAP_workflow _api.json", "r") as file_json:
            workflow = json.load(file_json)
        
        # 해상도 설정
        width, height = map(int, resolution.split('x'))
        workflow["45"]["inputs"].update({
            "width": width,
            "height": height
        })
        workflow["83"]["inputs"].update({
            "side_length": max(width, height)
        })
        
        # 성별 프롬프트 생성
        gender_prompt = get_gender_prompt(gender_option)
        
        # 프롬프트 결합 (성별 프롬프트를 맨 앞에 배치)
        age_prompt = age_prompts.get(age_option, "")
        final_prompt = f"{gender_prompt}, {BASE_PROMPT}"
        if age_prompt:
            final_prompt = f"{final_prompt}, {age_prompt}"
        if additional_prompt.strip():
            final_prompt = f"{final_prompt}, {additional_prompt}"
            
        final_negative = combine_prompts(BASE_NEGATIVE, additional_negative)
        
        # 시드 설정
        actual_seed = seed if seed_option == "fixed" else random.randint(1, 1500000)
        workflow["49"]["inputs"].update({
            "seed": actual_seed,
            "steps": steps,
            "denoise": denoise,
        })
        
        # 프롬프트 업데이트 - 노드 41번과 2번에 동일한 네거티브 프롬프트 적용
        workflow["42"]["inputs"]["text"] = final_prompt
        workflow["41"]["inputs"]["text"] = final_negative
        workflow["2"]["inputs"]["text"] = final_negative
        
        # 젠더 설정
        workflow["26"]["inputs"]["preset_expr"] = "#Female > #Male" if gender_option == "woman" else "#Female < #Male"
        
        # InstantID 파라미터 업데이트
        workflow["58"]["inputs"].update({
            "weight": instantid_weight,
            "start_at": instantid_start_at,
            "end_at": instantid_end_at
        })
        
        # PuLID 파라미터 업데이트
        workflow["75"]["inputs"].update({
            "weight": pulid_weight,
            "start_at": pulid_start_at,
            "end_at": pulid_end_at
        })
        
        # 페이스 마스크 파라미터 업데이트
        workflow["59"]["inputs"]["amount"] = face_mask_blur
        workflow["43"]["inputs"]["padding"] = face_mask_padding
        workflow["63"]["inputs"]["expand"] = face_mask_expand
        
        # 이미지 생성 시작
        start_time = time.time()
        start_queue(workflow)
        
        # 결과 대기
        comfy_output_path = wait_for_new_image(start_time)
        
        if comfy_output_path and os.path.exists(comfy_output_path):
            # 결과물 저장
            output_save_path = os.path.join(OUTPUT_SAVE_DIR, f"output_{timestamp}.png")
            shutil.copy(comfy_output_path, output_save_path)
            print(f"Saved output image to: {output_save_path}")
            print(f"Used seed value: {actual_seed}")
            
            # Gradio에 경로 반환
            return output_save_path
            
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
    requests.post(URL, data=data)

# Gradio 인터페이스 설정
with gr.Blocks(title="ID Photo Generator") as demo:
    gr.Markdown("# ID Photo Generator")
    gr.Markdown("Generate professional ID photos with customizable parameters.")
    
    with gr.Row():
        # 왼쪽 컬럼 - 기본 입력
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
                    value="노년"
                )
            
            # 프롬프트 입력
            additional_prompt = gr.Textbox(
                label="Additional Prompt (Optional)",
                placeholder="Add your custom prompt here...",
                value=""
            )
            additional_negative = gr.Textbox(
                label="Additional Negative Prompt (Optional)",
                placeholder="Add your custom negative prompt here...",
                value=""
            )
            
            # 기본 프롬프트 정보 표시
            with gr.Accordion("Default Prompts (Reference)", open=False):
                gr.Markdown(f"**Base Prompt:**\n{BASE_PROMPT}")
                gr.Markdown(f"**Base Negative Prompt:**\n{BASE_NEGATIVE}")
            
            resolution = gr.Radio(
                label="Output Resolution",
                choices=["1024x1024", "1216x832", "832x1216"],
                value="1024x1024"
            )
            
            steps = gr.Slider(label="Steps", minimum=1, maximum=100, step=1, value=10)
            denoise = gr.Slider(label="Denoise Strength", minimum=0.8, maximum=1, step=0.05, value=0.9)
            
            # Seed Controls를 기본 옵션으로 이동
            gr.Markdown("### Seed Controls")
            with gr.Row():
                seed_option = gr.Radio(
                    label="Seed Mode",
                    choices=["random", "fixed"],
                    value="random",
                    interactive=True
                )
                seed = gr.Number(
                    label="Fixed Seed Value", 
                    value=-1,
                    interactive=True,
                    visible=False
                )
            
        # 오른쪽 컬럼 - 고급 설정
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

    def update_seed_visibility(mode):
        return gr.update(visible=(mode == "fixed"))

    # Seed 모드 변경 이벤트 연결
    seed_option.change(
        fn=update_seed_visibility,
        inputs=[seed_option],
        outputs=[seed]
    )

    # Generate 버튼 및 이벤트 연결
    generate_btn = gr.Button("Generate", variant="primary", size="lg")
    generate_btn.click(
        fn=generate_image,
        inputs=[
            input_image, 
            additional_prompt,
            additional_negative,
            steps, denoise,
            instantid_weight, seed, seed_option,
            gender, age, resolution,
            instantid_start_at, instantid_end_at,
            pulid_weight, pulid_start_at, pulid_end_at,
            face_mask_blur, face_mask_padding, face_mask_expand
        ],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch(share=True, allowed_paths=[INPUT_SAVE_DIR, OUTPUT_SAVE_DIR])