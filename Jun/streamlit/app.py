import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_ollama import ChatOllama

def extract_text_between_newlines(text):
    """
    주어진 문자열에서 첫 번째 \n과 두 번째 \n 사이의 텍스트를 추출합니다.
    \n이 없을 경우 전체 텍스트를 반환합니다.

    Args:
        text (str): 입력 문자열.

    Returns:
        str: 첫 번째와 두 번째 \n 사이의 텍스트, 또는 \n이 없으면 입력 텍스트 그대로 반환.
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
    # 초기화 버튼
    clear_button = st.button("대화 초기화")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# LLM 설정
llm = ChatOllama(
    model="llama3-couple-v8:latest",
    temperature=0.1,
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