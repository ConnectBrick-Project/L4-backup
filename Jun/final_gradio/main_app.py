import gradio as gr
from main_interface_kor import create_main_interface

# from img2img import img2img_demo  
from llava_interface import create_llava_interface



# 필요한 디렉토리 경로 설정
INPUT_SAVE_DIR = "/home/eardream2/Jun/input"   
OUTPUT_SAVE_DIR = "/home/eardream2/Jun/output"
SAVED_IMAGES_DIR = "/home/eardream2/Jun/saved"

# 메인 애플리케이션 생성
demo = gr.TabbedInterface(
    interface_list=[
        create_main_interface(),  # 메인 인터페이스
        create_llava_interface(),
        # img2img_demo,            # img2img 인터페이스
    ],
    tab_names=[
        "기본", 
        "이미지 캡션",
        # "Image to Image"
    ]
)

# 애플리케이션 실행
if __name__ == "__main__":
    demo.launch(
        share=True,
        allowed_paths=[
            INPUT_SAVE_DIR,
            OUTPUT_SAVE_DIR,
            SAVED_IMAGES_DIR,
            "/home/eardream2/ComfyUI/output"
        ]
    )
