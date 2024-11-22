# 베이스 이미지 설정 (Python 3.10 이미지 사용)
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# ComfyUI 설치 (GitHub에서 소스코드를 가져옴)
RUN git clone https://github.com/comfyanonymous/ComfyUI.git .

# 파이썬 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 호스트와 컨테이너 사이의 포트 연결 (ComfyUI는 기본적으로 8188 포트를 사용)
EXPOSE 8188

# 컨테이너 실행 시 실행할 명령어
CMD ["python", "main.py"]

