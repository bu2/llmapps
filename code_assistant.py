import streamlit as st
import ollama


LLMs = [
    'phi3',
    'mistral',
    'qwen2',
    'llama3.1',
    'gemma2',
    'mistral-nemo',
    'phi3:14b',
    'codestral',
    'gemma2:27b',
    'mixtral',
    'llama3.1:70b',
    'qwen2:72b',
    'qwen:110b',
    'mistral-large',
    'mixtral:8x22b',
]

LLM = 'llama3.1'

SYSTEM = 'Output only code with comments.'

    
def llm(prompt):
    return ollama.generate(prompt=prompt, model=LLM, stream=True)


def streaming_callback(x):
    for chunk in x:
        yield chunk['response']

        
st.set_page_config(layout="wide")    
st.title('Code Assistant')
st.markdown('##### Ask the LLM to generate some code, then iterate on it with successive prompts.')

LLM = st.selectbox('LLM:', LLMs, LLMs.index(LLM))


st.session_state.code = st.session_state.get('code', '')

if st.session_state.code == '':
    st.session_state.prompt_label = 'Ask for code...'
elif st.session_state.code != '':
    st.markdown(st.session_state.code)
    st.session_state.prompt_label = 'Iterate on the code...'

if st.button('Edit...'):
    st.text_area('Code:', key='code', height=600)
    
if prompt := st.chat_input(st.session_state.prompt_label):
    response = llm(st.session_state.code + '\n\n\nPrompt: ' + prompt + '\n\n\nSystem: ' + SYSTEM)
    code = st.write_stream(streaming_callback(response))
    st.session_state.code = code
    st.rerun()
