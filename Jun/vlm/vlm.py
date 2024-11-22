import streamlit as st
from vllm import LLM, SamplingParams
import time

st.title("MZ Talk ChatBot")

# Sidebar configurations
with st.sidebar:
    clear_button = st.button("대화 초기화")
    
    # 고정된 모델 사용
    model = "junn991/mz_talk"
    st.info(f'사용 모델: {model}')
    
    # vLLM specific configurations
    num_gpus = st.slider("GPU 수", min_value=1, max_value=8, value=1)
    temperature = st.slider("Temperature(높을수록 더 창의적인 답변)", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    max_tokens = st.slider("Max Tokens(출력 길이)", min_value=64, max_value=2048, value=512)
    
    st.divider()
    st.caption("Made with vLLM 🚀")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "llm" not in st.session_state:
    st.session_state["llm"] = None

def initialize_vllm():
    """Initialize vLLM model with current settings"""
    if st.session_state["llm"] is None:
        with st.spinner("MZ Talk 모델을 로딩중입니다... ⏳"):
            try:
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=0.95,
                    max_tokens=max_tokens,
                    stop=['<|endoftext|>', '</s>', '<|im_end|>', '<|end_of_text|>']
                )
                
                llm = LLM(
                    model=model,
                    tensor_parallel_size=num_gpus,
                    trust_remote_code=True,
                    max_model_len=max_tokens,
                    gpu_memory_utilization=0.8  # GPU 메모리 사용률 설정
                )
                
                st.session_state["llm"] = (llm, sampling_params)
                st.success("모델 로딩 완료! 채팅을 시작해보세요 😊")
            except Exception as e:
                st.error(f"모델 로딩 중 오류가 발생했습니다: {str(e)}")

def generate_response(prompt):
    """Generate response using vLLM"""
    llm, sampling_params = st.session_state["llm"]
    try:
        outputs = llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()
    except Exception as e:
        return f"응답 생성 중 오류가 발생했습니다: {str(e)}"

def display_messages():
    """Display all messages in the chat"""
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
            st.write(message["content"])

# Clear chat history if button is clicked
if clear_button:
    st.session_state["messages"] = []
    st.success("대화가 초기화되었습니다!")

# Display chat history
display_messages()

# Initialize vLLM if not already initialized
if st.session_state["llm"] is None:
    initialize_vllm()

# Chat input
if user_input := st.chat_input("MZ Talk과 대화해보세요..."):
    # Display user message
    st.chat_message("user", avatar="🧑‍💻").write(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    # Generate and display response
    with st.chat_message("assistant", avatar="🤖"):
        if st.session_state["llm"] is not None:
            with st.spinner("생각하는 중..."):
                response = generate_response(user_input)
                st.write(response)
                st.session_state["messages"].append({"role": "assistant", "content": response})
        else:
            st.error("모델이 초기화되지 않았습니다. 잠시 후 다시 시도해주세요.")