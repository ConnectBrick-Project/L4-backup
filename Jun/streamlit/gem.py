import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 직접 모델 로딩
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

fine_tuned_model_path = "/home/eardream2/Jun/Fine_TT/gemma/gemma2-2b-it-sft-couple-merged"
model = AutoModelForCausalLM.from_pretrained(
    fine_tuned_model_path,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)

def convert_to_gemma_format(instruction):
    return (
        f"Below is an instruction that describes a questions in casual conversation. Write an appropriate response that follows the instruction.\n"
        f"{tokenizer.bos_token}<start_of_turn>user\n"
        f"{instruction}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

def generate_response(instruction_str):
    inputs = tokenizer(
        [convert_to_gemma_format(instruction_str)],
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=True,
        do_sample=True,
        temperature=0.5,
        top_p=0.95
    )
    response = tokenizer.batch_decode(outputs)[0]
    clean_response = response.split("<start_of_turn>model\n")[1].replace("<end_of_turn>", "").strip()
    return clean_response

st.title("Gemma 챗봇")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("메시지를 입력하세요"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    response = generate_response(prompt)
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})