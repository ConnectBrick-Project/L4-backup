class Config:
    PAGE_TITLE = "Streamlit Ollama Chatbot"

    OLLAMA_MODELS = ('llama3-8b-stt-unsloth', 'codellama:13b', 'llama2-uncensored', 
                    'llama2:7b', 'llama2:13b', 'mistral', 'mixtral')

    SYSTEM_PROMPT = f"""It's a casual conversation between a son (user) in his 20s and his dad (assistant) in his 50s, 
        and the dad (assistant) needs to continue the casual conversation by answering the son's (user) questions in Korean."""
    