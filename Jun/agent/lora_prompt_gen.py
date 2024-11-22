import os
import base64
from groq import Groq
import json
import re
from pathlib import Path
from typing import List, Dict, Generator, Optional
import time
import shutil

from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()

from pydantic import BaseModel, Field
from typing import List

class ImageTags(BaseModel):
    detailed_tags: List[str] = Field(
        description="List of detailed tags describing the image, including quality, subject features, pose, clothing, and atmosphere"
    )

# Pydantic 모델 정의
class ImageProcessor:
    def __init__(self, groq_client):
        self.client = groq_client
        self.parser = JsonOutputParser(pydantic_object=ImageTags)

        self.prompt = PromptTemplate(
            template="""Analyze the given image and generate essential tags for LoRA training. Focus only on the most distinctive and important characteristics.

Structured guidelines for tag generation:
1. Essential Character Features (Keep minimal but precise):
   - Gender and exact age (e.g., "male", "30s")
   - Key facial features only (most distinctive 2-3 features)
   - Distinctive hair style and color
   - Notable expression if any
   - Smiling if it makes you feel happy

2. Core Style Elements:
   - Basic pose description (e.g., "frontal view", "side view")
   - Main clothing type (just primary items)
   - Dominant color or tone
   - Overall mood/atmosphere (1-2 words)

REQUIRED Elements:
- Age must be specified (e.g., "20s", "30s", "40s", etc.)
- Keep total tags under 15
- Use simple, clear descriptions
- Focus on unique identifying features only

        {format_instructions}""",
            input_variables=[],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        self.analysis_prompt = self.prompt.format()

    def analyze_image(self, base64_image: str) -> str:
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.analysis_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }],
                model='llama-3.2-11b-vision-preview',
                temperature=0.3  # 더 일관된 출력을 위해 temperature 낮춤
            )

            response_text = chat_completion.choices[0].message.content.strip()

            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed_data = json.loads(json_str)
                    if 'detailed_tags' in parsed_data and isinstance(parsed_data['detailed_tags'], list):
                        return json.dumps(parsed_data, ensure_ascii=False)
                    else:
                        raise ValueError("Invalid JSON structure")
                except Exception as parse_error:
                    print(f"JSON parsing error: {str(parse_error)}")
                    raise
            else:
                raise ValueError("No JSON object found in response")

        except Exception as e:
            print(f"API Error: {str(e)}")
            raise

    @staticmethod
    def clean_json_response(response: str) -> str:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON object found in response")

        json_str = json_match.group(0)
        json_str = json_str.encode('utf-8', 'ignore').decode('utf-8')
        json_str = json_str.rstrip('."')
        json_str = json_str.replace('\n', ' ').replace('\r', '')
        json_str = json_str.replace('\\"', "'")
        json_str = re.sub(r'(?<=\w)"(?=\w)', '', json_str)
        json_str = re.sub(r',\s*}', '}', json_str)

        return json_str

    @staticmethod
    def encode_image(image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def save_prompt_file(image_path: str, tags: List[str], output_dir: str = None) -> str:
        image_path = Path(image_path)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = output_dir / f"{image_path.stem}.txt"
        else:
            prompt_path = image_path.with_suffix('.txt')
        tag_content = ', '.join(tags)
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(tag_content)
        return str(prompt_path)

class FileManager:
    @staticmethod
    def get_unprocessed_files(input_dir: str, output_dir: str) -> List[Path]:
        """입력 폴더의 이미지 파일 중 출력 폴더에 대응되는 txt 파일이 없는 것들만 반환"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        input_files = [f for f in input_path.iterdir()
                      if f.is_file() and f.suffix.lower() in image_extensions]

        unprocessed = [f for f in input_files
                      if not (output_path / f"{f.stem}.txt").exists()]

        return unprocessed

class TagReviewAgent:
    def __init__(self, groq_client):
        self.client = groq_client
        self.review_prompt = """

        Please enhance the following tags to make them more natural

        Current tags: {tags}

        Guidelines:
        1. Add descriptive tags for style and atmosphere
        2. Order tags from general to specific
        3. Add any missing essential features
        4. **REQUIRED TAGS**: MUST include "portrait", "korean", "frontal view" for front related words, and "smile"

        Return only the comma-separated tags.)
        """

    def enhance_tags(self, file_path: Path) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                current_tags = f.read().strip()

            completion = self.client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": self.review_prompt.format(tags=current_tags)
                }],
                model="llama-3.2-90b-text-preview" # llama-3.2-90b-text-preview /
            )

            return completion.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error enhancing tags for {file_path.name}: {str(e)}")
            return current_tags

def process_directory(dir_path: str, output_dir: str) -> Generator[str, None, None]:
    """한 디렉토리의 모든 이미지 처리"""
    groq_client = Groq()
    processor = ImageProcessor(groq_client)
    file_manager = FileManager()

    unprocessed = file_manager.get_unprocessed_files(dir_path, output_dir)

    if not unprocessed:
        yield "All files are already processed."
        return

    yield f"\nFound {len(unprocessed)} files to process"

    success_count = 0
    error_count = 0
    for image_path in unprocessed:
        try:
            base64_image = processor.encode_image(str(image_path))
            raw_response = processor.analyze_image(base64_image)

            try:
                cleaned_response = processor.clean_json_response(raw_response)
                response_data = json.loads(cleaned_response)

                if 'detailed_tags' in response_data:
                    processor.save_prompt_file(
                        image_path,
                        response_data['detailed_tags'],
                        output_dir
                    )
                    yield f"\nSuccessfully created tags for: {image_path.name}"
                    success_count += 1
                else:
                    yield f"\nInvalid response format for: {image_path.name}"
                    error_count += 1

            except (json.JSONDecodeError, ValueError) as e:
                yield f"\nError parsing response for {image_path.name}: {str(e)}"
                error_count += 1

            time.sleep(1)

        except Exception as e:
            yield f"\nError processing {image_path.name}: {str(e)}"
            error_count += 1

    yield f"\nProcessed {success_count} files successfully, {error_count} errors."

def process_tag_files(output_dir: str) -> Generator[str, None, None]:
    """출력 디렉토리의 태그 파일들을 직접 처리하는 함수"""
    groq_client = Groq()
    reviewer = TagReviewAgent(groq_client)
    output_path = Path(output_dir)

    tag_files = list(output_path.glob("*.txt"))

    if not tag_files:
        yield "No tag files found for review."
        return

    yield f"\nFound {len(tag_files)} tag files to review"

    success_count = 0
    error_count = 0

    for tag_file in tag_files:
        try:
            # 태그 개선
            enhanced_tags = reviewer.enhance_tags(tag_file)

            # 개선된 태그로 직접 덮어쓰기
            with open(tag_file, 'w', encoding='utf-8') as f:
                f.write(enhanced_tags)

            yield f"\nEnhanced tags for: {tag_file.name}"
            success_count += 1

        except Exception as e:
            yield f"\nError processing {tag_file.name}: {str(e)}"
            error_count += 1

        time.sleep(1)

    yield f"\nReview Summary: {success_count} files enhanced, {error_count} errors"

def chat_stream(input_dir: str, output_dir: str, max_cycles: int = 5) -> Generator[str, None, None]:
    """메인 처리 흐름 - 이미지 처리 후 태그 리뷰 수행"""
    yield "\n=== Starting image processing ==="

    for cycle in range(max_cycles):
        yield f"\n--- Cycle {cycle + 1}/{max_cycles} ---"

        file_manager = FileManager()
        unprocessed = file_manager.get_unprocessed_files(input_dir, output_dir)

        if not unprocessed:
            yield "\nAll files are processed!"
            break

        yield f"\nFound {len(unprocessed)} files to process"

        for progress in process_directory(input_dir, output_dir):
            yield progress

        if cycle < max_cycles - 1:
            remaining = file_manager.get_unprocessed_files(input_dir, output_dir)
            if remaining:
                yield f"\nStill have {len(remaining)} unprocessed files. Moving to next cycle..."
            else:
                yield "\nAll files processed successfully!"
                break

    # 태그 리뷰 프로세스 시작
    yield "\n=== Starting tag review process ==="

    for review_progress in process_tag_files(output_dir):
        yield review_progress

    yield "\n=== Tag review process complete ==="

if __name__ == "__main__":
    # 사용자에게 입력 받음
    # input_dir = input("Enter the path to your image folder: ")
    # output_dir = input("Enter output directory for tag files: ")

    # 미리 입력
    input_dir = "/content/input"
    output_dir = "/content/output5"

    print("\nProcessing images and reviewing tags...")
    for message in chat_stream(input_dir, output_dir):
        print(message, end="", flush=True)