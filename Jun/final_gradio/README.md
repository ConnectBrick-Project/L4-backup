# 사용법
0. comfyui on
1. conda activate jun_web
2. python main_app.py


# 되는 파일.
- main_app
- main_interface_kor
- llava_interface

# 주의
- workflow파일 base쓰면 전부 여성화되버림 -> 폐기
- 원래꺼 써야함




----------------------새로운 기능을 추가하려면--------------------------------

새로운 파일 생성 (예: new_feature.py)
인터페이스 함수 구현
app.py에서 import하고 interface_list와 tab_names에 추가

예를 들어 새로운 기능을 추가할 때:
# new_feature.py
def create_new_feature():
    with gr.Blocks() as new_interface:
        # 새로운 기능의 UI 구현
        pass
    return new_interface

# app.py에 추가
from new_feature import create_new_feature

demo = gr.TabbedInterface(
    interface_list=[
        create_main_interface(),
        img2img_demo,
        txt_to_img,
        create_new_feature()    # 새로운 기능 추가
    ],
    tab_names=[
        "ID Photo Generator",
        "Image to Image",
        "Text to Image",
        "New Feature"          # 새로운 탭 이름
    ]
)


# main_interface
## v2 : 기능추가
성별에 따른 의상 프롬프트 추가

남성 선택 시 "formal suit" 추가
여성 선택 시 "women suit" 추가


주의사항 섹션 추가

4가지 주요 주의사항을 마크다운 형식으로 표시


UI 한국어화

모든 레이블과 버튼 텍스트를 한국어로 변경
성별 선택 옵션을 "남성"/"여성"으로 변경
시드 모드 옵션을 "랜덤"/"고정"으로 변경