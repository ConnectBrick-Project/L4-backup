import gradio as gr
import os
import json
import requests
import time

URL = "http://localhost:5001/api/prompt"
OUTPUT_DIR = "/home/eardream2/ComfyUI/output"

def get_latest_image(folder):
    files = os.listdir(folder) # 지정한 폴더의 모든 파일 확인
    image_files = [f for f in files if f.lower().endswith(('.png','.jpg','.jpeg'))]
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    latest_image = os.path.join(folder, image_files[-1]) if image_files else None
    return latest_image



def start_queue(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    requests.post(URL, data=data)



def generate_image(prompt_text, step_count):
    with open("img2img.json", "r") as file_json:
        prompt = json.load(file_json)
        prompt["6"]["inputs"]["text"] = f"digital artwork of a {prompt_text}"
        prompt["3"]["inputs"]["steps"] = step_count
    
    previous_image = get_latest_image(OUTPUT_DIR)

    start_queue(prompt)


    while True:
        latest_image = get_latest_image(OUTPUT_DIR)
        if latest_image != previous_image:
            return latest_image

        time.sleep(1)

demo = gr.Interface(fn=generate_image, inputs=["image"], outputs=["image"])

demo.launch()