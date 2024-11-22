import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_ollama import ChatOllama
import re

def preprocess_text(text):
    """
    텍스트를 전처리하는 함수
    1. Human과 Assistant 텍스트를 기준으로 대화를 분리
    2. 불필요한 공백 제거
    """
    # Human과 Assistant 텍스트를 기준으로 분리
    conversations = re.split(r'(Human|Assistant)', text)
    # 빈 문자열 제거
    conversations = [conv.strip() for conv in conversations if conv.strip()]
    
    processed_messages = []
    current_role = None
    current_content = []
    
    for item in conversations:
        if item in ['Human', 'Assistant']:
            if current_role and current_content:
                processed_messages.append({
                    'role': current_role.lower(),
                    'content': ' '.join(current_content)
                })
                current_content = []
            current_role = item
        else:
            current_content.append(item)
    
    # 마지막 메시지 처리
    if current_role and current_content:
        processed_messages.append({
            'role': current_role.lower(),
            'content': ' '.join(current_content)
        })
    
    return processed_messages

def extract_text_between_newlines(text):
    """
    주어진 문자열에서 첫 번째 \n과 두 번째 \n 사이의 텍스트를 추출합니다.
    \n이 없을 경우 전체 텍스트를 반환합니다.
    """
    first_newline = text.find('\n')
    
    if first_newline == -1:
        return text.strip()
    
    second_newline = text.find('\n', first_newline + 1)
    
    if second_newline != -1:
        return text[first_newline + 1:second_newline].strip()
    else:
        return text[first_newline + 1:].strip()

# 페이지 제목
st.title("Simple ChatBot based on Ollama")

# 사이드바 설정
with st.sidebar:
    clear_button = st.button("대화 초기화")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# LLM 설정
llm = ChatOllama(
    model="llama3-couple-v6:latest",
    temperature=1,
)

# 대화기록 저장 함수
def add_message(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))

# 초기화 버튼이 눌리면 대화기록 초기화
if clear_button:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
for message in st.session_state["messages"]:
    st.chat_message(message.role).write(message.content)

# 사용자 입력
user_input = st.chat_input("대화를 입력하세요")

if user_input:
    # 입력된 텍스트 전처리
    if "Human" in user_input or "Assistant" in user_input:
        processed_messages = preprocess_text(user_input)
        # 전처리된 메시지들을 세션 상태에 추가
        for msg in processed_messages:
            add_message(msg['role'], msg['content'])
    else:
        # 일반적인 사용자 입력 처리
        messages = [{"role": msg.role, "content": msg.content} 
                    for msg in st.session_state["messages"]]
        messages.append({"role": "human", "content": user_input})
        
        with st.chat_message("assistant"):
            container = st.empty()
            response = llm.stream(input=messages)
            answer = ""
            for token in response:
                token_content = extract_text_between_newlines(token.content)
                answer += token_content
                container.markdown(answer)
        
        # 대화 기록 저장
        add_message("human", user_input)
        add_message("assistant", answer)