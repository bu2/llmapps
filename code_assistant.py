import ollama
import streamlit as st


LLMs = [
    'qwen2.5:3b',
    'llama3.2:3b',
    'falcon3:3b',
    'phi3.5',
    'yi',
    'mistral',
    'falcon3:7b',
    'llama3.1',
    'qwen2.5',
    'qwen2.5-coder',
    'opencoder',
    'tulu3',
    'falcon3:7b',
    'deepseek-r1:7b',
    'deepseek-r1:8b',
    'granite3.1-dense',
    'dolphin3',
    'command-r7b',
    'gemma2',
    'falcon3:10b',
    'mistral-nemo',
    'phi3:14b',
    'qwen2.5:14b',
    'qwen2.5-coder:14b',
    'deepseek-r1:14b',
    'codestral',
    'mistral-small',
    'gemma2:27b',
    'command-r',
    'qwen2.5:32b',
    'qwen2.5-coder:32b',
    'qwq',
    'deepseek-r1:32b',
    'mixtral',
    'llama3.1:70b',
    'llama3.3:70b',
    'nemotron',
    'tulu3:70b',
    'deepseek-r1:70b',
    'qwen2.5:72b',
    'command-r-plus',
    'qwen:110b',
    'mistral-large',
    'mixtral:8x22b',
]

LLM = 'deepseek-r1:70b'

SYSTEM = 'Output only code with comments.'

    
def llm(prompt):
    return ollama.generate(prompt=prompt, model=LLM, options=dict(temperature=0), stream=True)


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
