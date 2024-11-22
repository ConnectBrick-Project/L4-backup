import os
import gradio as gr
import ollama
from datetime import datetime
from PIL import Image
import json

# Constants
INPUT_DIR = "/home/eardream2/Jun/input"   
OUTPUT_DIR = "/home/eardream2/Jun/output"
SAVED_DIR = "/home/eardream2/Jun/saved"

def ensure_dirs():
    """Ensure required directories exist"""
    for dir in [INPUT_DIR, OUTPUT_DIR, SAVED_DIR]:
        os.makedirs(dir, exist_ok=True)

def process_image(input_image, focus_area=""):
    """Process image to generate focused description and extract balanced tags"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = os.path.join(INPUT_DIR, f"input_{timestamp}.png")
        
        if input_image is not None:
            Image.fromarray(input_image).save(input_path)
        else:
            raise ValueError("No image provided")

        # Modify prompt based on focus area
        if focus_area:
            base_prompt = (
                f"Describe this image with particular attention to {focus_area}. "
                f"Provide clear details about {focus_area} aspects, including specific characteristics "
                f"and notable features related to {focus_area}."
            )
        else:
            base_prompt = (
                "Provide a clear description of this image, focusing on distinctive visual elements "
                "and notable features. Be specific but concise."
            )
        
        # Get focused description
        description_response = ollama.chat(
            model="llava",
            messages=[{
                'role': 'user',
                'content': base_prompt,
                'images': [input_path]
            }]
        )
        
        description = description_response['message']['content']
        
        # Extract balanced tags from the description
        if focus_area:
            tag_prompt = (
                f"From this description, extract exactly 5 clear and specific phrases about {focus_area}. "
                f"Each phrase should combine 2-3 words to create a meaningful description. "
                f"Examples formats:\n"
                f"- 'warm smile with crinkled eyes'\n"
                f"- 'vintage denim jacket'\n"
                f"- 'soft pastel colors'\n"
                f"- 'bustling city street'\n"
                f"Use only information from the description. Return just the phrases separated by commas."
            )
        else:
            tag_prompt = (
                "From this description, extract 5 clear and specific phrases about key elements. "
                "Each phrase should combine 2-3 words to create a meaningful description. "
                "Examples: 'warm smile with crinkled eyes', 'vintage denim jacket', 'modern glass building'. "
                "Use only information from the description. Return just the phrases separated by commas."
            )

        tag_response = ollama.chat(
            model="llava",
            messages=[{
                'role': 'user',
                'content': tag_prompt,
                'context': description
            }]
        )
        
        # Clean and format tags
        content = tag_response['message']['content'].strip()
        content = (content.replace('{', '').replace('}', '')
                         .replace('"', '').replace(':', '')
                         .replace('[', '').replace(']', '')
                         .replace('\n', ', '))
        
        # Split and clean tags
        tags = [tag.strip() for tag in content.split(',') if tag.strip()]
        tags = [tag.replace('_', ' ') for tag in tags][:5]
        
        return description, input_path, ", ".join(tags)

    except Exception as e:
        raise gr.Error(f"Error processing image: {str(e)}")

def save_image_and_caption(image_path, caption, tags, custom_filename=""):
    """Save the processed image, caption, and tags"""
    try:
        if not custom_filename:
            custom_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_dir = os.path.join(SAVED_DIR, custom_filename)
        os.makedirs(save_dir, exist_ok=True)

        # Save image
        try:
            if isinstance(image_path, str) and os.path.isfile(image_path):
                with Image.open(image_path) as img:
                    img.save(os.path.join(save_dir, f"{custom_filename}.png"))
            else:
                Image.fromarray(image_path).save(os.path.join(save_dir, f"{custom_filename}.png"))
        except Exception as e:
            return f"Error saving image: {str(e)}"

        # Save caption and tags
        try:
            with open(os.path.join(save_dir, f"{custom_filename}.txt"), "w", encoding="utf-8") as f:
                f.write(f"Description:\n{caption}\n\nTags:\n{tags}")
        except Exception as e:
            return f"Error saving text: {str(e)}"

        return f"Saved as {custom_filename}"
    except Exception as e:
        return f"Save failed: {str(e)}"

def create_llava_interface():
    """Create the Gradio interface with Korean UI"""
    with gr.Blocks() as llava_interface:
        gr.Markdown("# 이미지 설명 및 태그 생성기")
        gr.Markdown("이미지를 업로드하면 상세한 설명과 태그를 생성합니다. 원하는 경우 특정 부분에 초점을 맞출 수 있습니다.")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="이미지 업로드",
                    type="numpy"
                )
                focus_area = gr.Textbox(
                    label="초점 영역 (선택사항)",
                    placeholder="분석하고 싶은 특정 부분을 입력하세요 (예: 색상, 사물, 감정, 스타일)"
                )
                filename = gr.Textbox(
                    label="저장할 파일명 (선택사항)",
                    placeholder="파일명을 입력하세요 (확장자 제외)"
                )

            with gr.Column():
                output_text = gr.Textbox(
                    label="생성된 설명",
                    interactive=False,
                    lines=10
                )
                tags_text = gr.Textbox(
                    label="생성된 태그",
                    interactive=False
                )
                status = gr.Textbox(
                    label="저장 상태",
                    interactive=False
                )

        with gr.Row():
            process_btn = gr.Button("설명 및 태그 생성", variant="primary")
            save_btn = gr.Button("이미지, 설명, 태그 저장")
        
        # Set up event handlers
        process_btn.click(
            fn=process_image,
            inputs=[input_image, focus_area],
            outputs=[output_text, input_image, tags_text]
        )

        save_btn.click(
            fn=save_image_and_caption,
            inputs=[input_image, output_text, tags_text, filename],
            outputs=[status]
        )

    return llava_interface

if __name__ == "__main__":
    ensure_dirs()
    demo = create_llava_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[INPUT_DIR, OUTPUT_DIR, SAVED_DIR]
    )