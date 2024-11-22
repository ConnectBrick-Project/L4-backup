import streamlit as st
import requests
import json
from transformers import AutoTokenizer

# 페이지 설정
st.set_page_config(page_title="AI 챗봇", page_icon="💝")
st.title("AI 챗봇")

# Initialize tokenizer
@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("beomi/Llama-3-Open-Ko-8B")

tokenizer = load_tokenizer()

# vLLM 서버 설정
vllm_host = "http://localhost:8000"
headers = {"Content-Type": "application/json"}

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "당신은 질문에 친근하게 답변하는 애인입니다."}
    ]

def clean_response(response):
    """첫 번째 의미 있는 응답만 추출"""
    if not response:
        return ""
        
    # 특정 태그나 텍스트 이전의 내용 제거
    for tag in ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]:
        if tag in response:
            response = response.split(tag)[-1]
            
    # 첫 번째 문장 추출 (마침표, 물음표, 느낌표 기준)
    sentences = response.replace('?', '.').replace('!', '.').split('.')
    for sentence in sentences:
        clean_text = sentence.strip()
        if clean_text:  # 의미 있는 첫 번째 문장 반환
            return clean_text
            
    return response.strip()

def generate_response(messages):
    try:
        # 토크나이저로 채팅 템플릿 적용
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # vLLM API 호출
        response = requests.post(
            f"{vllm_host}/generate",
            headers=headers,
            json={
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": 0.9,
                "stop": ["<|end_of_text|>"]
            }
        )
        
        if response.status_code == 200:
            response_text = response.json()["text"][0]
            return clean_response(response_text)
            
    except Exception as e:
        st.error(f"오류 발생: {str(e)}")
        return None

# Sidebar for parameters
with st.sidebar:
    st.header("생성 파라미터")
    max_tokens = st.slider("최대 토큰 수", 1, 1024, 512)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

# Chat container for better scrolling
chat_container = st.container()

# Display chat history in the container
with chat_container:
    for message in st.session_state.messages[1:]:  # Skip system message
        role = "🧑" if message["role"] == "user" else "💝"
        st.markdown(f"**{role}:** {message['content']}")

# Chat input
with st.form(key="chat_form"):
    user_input = st.text_input("메시지를 입력하세요:", key="user_message")
    submit = st.form_submit_button("전송")
    
    if submit and user_input:
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 응답 생성
        response = generate_response(st.session_state.messages)
        
        if response:
            # 어시스턴트 메시지 추가
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        st.rerun()

# Clear chat button    
if st.sidebar.button("대화내용 지우기"):
    st.session_state.messages = [
        {"role": "system", "content": "당신은 질문에 친근하게 답변하는 애인입니다."}
    ]
    st.rerun()