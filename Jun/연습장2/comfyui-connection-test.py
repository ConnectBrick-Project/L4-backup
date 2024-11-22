import gradio as gr
import requests
import websocket
import json
import uuid
import time

SERVER_ADDRESS = "35.233.216.129:5001"
client_id = str(uuid.uuid4())

def test_connections():
    results = []
    
    # 1. HTTP 연결 테스트
    try:
        response = requests.get(f"http://{SERVER_ADDRESS}/")
        results.append(f"HTTP 기본 연결: 성공 (상태 코드: {response.status_code})")
    except Exception as e:
        results.append(f"HTTP 기본 연결: 실패 ({str(e)})")
    
    # 2. System Stats 테스트
    try:
        response = requests.get(f"http://{SERVER_ADDRESS}/system_stats")
        if response.status_code == 200:
            results.append("System Stats API: 성공")
            results.append(f"시스템 정보: {json.dumps(response.json(), indent=2)}")
        else:
            results.append(f"System Stats API: 실패 (상태 코드: {response.status_code})")
    except Exception as e:
        results.append(f"System Stats API: 실패 ({str(e)})")
    
    # 3. WebSocket 연결 테스트 (타임아웃 설정)
    try:
        ws = websocket.WebSocket()
        ws.settimeout(5)  # 5초 타임아웃 설정
        ws_url = f"ws://{SERVER_ADDRESS}/ws?clientId={client_id}"
        results.append(f"WebSocket URL: {ws_url}")
        ws.connect(ws_url)
        results.append("WebSocket 연결: 성공")
        ws.close()
    except Exception as e:
        results.append(f"WebSocket 연결: 실패 ({str(e)})")
        results.append("가능한 원인:")
        results.append("1. GCP 방화벽에서 5001 포트가 차단됨")
        results.append("2. ComfyUI 서버의 WebSocket 설정 문제")
        results.append("3. 네트워크 라우팅 문제")
    
    return "\n".join(results)

demo = gr.Interface(
    fn=test_connections,
    inputs=[],
    outputs=gr.Textbox(label="연결 테스트 결과"),
    title="ComfyUI 상세 연결 테스트",
    description="HTTP와 WebSocket 연결을 상세하게 테스트합니다."
)

if __name__ == "__main__":
    demo.launch(share=True)