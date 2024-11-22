import gradio as gr
from pathlib import Path
import json
from PIL import Image


from websockets_api import get_prompt_images
from settings import COMFY_UI_PATH

from img2img import img2img # 이게 어떻게 되는거지, 함수도 아닌데
from img2img2 import img2img2


def process(positive ,img, slider):
    with open("/home/eardream2/Jun/연습장2/basic_workflow.json", "r", encoding="utf-8") as f:
        prompt = json.load(f) # workflow파일(json)을 python dict 형태로 load

    prompt["6"]["inputs"]["text"] = f"a portrait of a {positive}, highly detail, high resolution"
    # 다른 값 조절하고 싶으면 위에꺼 복사하고, 노드id 찾아서 조정할 파라미터 지정 ㅇㅇ
    #prompt["6"]["inputs"]["weight"] = slider

    images = get_prompt_images(prompt) # prompt 이미지를 가져오고 이미지 반환
    return images



txt_to_img = gr.Interface(
    fn=process,
    inputs=[gr.Textbox(label="Positive Prompt: ")],
    outputs=[gr.Gallery(label="Outputs: ")]
)


# 여러 페이지(tab) 기능
demo = gr.TabbedInterface(interface_list=[txt_to_img,img2img,img2img2], tab_names=["txt_to_img workflow", "img2img workflow","3"])




demo.queue()
    # 사용자 : 대기열에서 자신의 순서 알 수 있음
    # 개발자 : 허용되는 최대 제한 수 설정 가능
        # Ex : demo.queue(max_size=20)
demo.launch() # 앱 시작