import ollama
import streamlit as st

LLMs = [
    'llama3.2:3b',
    'qwen2.5:3b',
    'falcon3:3b',
    
    # 'phi',
    # 'phi3',
    'phi3.5',
    
    # 'llama2',
    
    'mistral',
    # 'mathstral',

    # 'qwen2',
    # 'qwen2-math',
    'qwen2.5',
    'qwen2.5-coder',
    
    # 'internlm2',
    
    # 'llama3',
    'llama3.1',
    # 'hermes3',
    'tulu3',
    'falcon3:7b',

    'deepseek-r1:7b',
    'deepseek-r1:8b',

    'granite3.1-dense',
    'dolphin3',

    # 'aya',
    # 'yi-coder',

    'granite3.1-dense',

    'opencoder',

    'command-r7b',

    # 'gemma',
    'gemma2',

    # 'glm4',
    # 'codegeex4',

    'falcon3:10b',
    
    'mistral-nemo',

    'phi3:14b',
    'qwen2.5:14b',
    'qwen2.5-coder:14b',
    'deepseek-r1:14b',

    # 'deepseek-v2',
    # 'deepseek-coder-v2',
    
    # 'starcoder2:15b',

    'mistral-small',

    'codestral',

    'gemma2:27b',

    'command-r',

    'qwen2.5:32b',
    'qwen2.5-coder:32b',
    'qwq',

    'deepseek-r1:32b',
    
    'deepseek-coder:33b',

    # 'granite-code:34b',
    
    'yi:34b',
    'aya:35b',

    'mixtral',
    
    'llama3:70b',
    'llama3.1:70b',
    'llama3.3:70b',
    'hermes3:70b',
    'nemotron',

    'deepseek-r1:70b',

    'qwen2:72b',
    'qwen2-math:72b',
    'qwen2.5:72b',

    'command-r-plus',

    'qwen:110b',

    'mistral-large',
    
    'mixtral:8x22b',
]
TEMPERATURE = 0
JUDGE = 'chatgpt'

EVAL_PROMPT = '''I asked the following question to different LLMs:

%s

And here are their answers:

%s


Rate all answers on a scale from 0 to 100.
Which LLM did provide the best answer ?
Justify your choice.
If there is no good answer, say it explicitely.'''

def llm(prompt, model, temperature=TEMPERATURE):
    return ollama.generate(model=model, prompt=prompt, options=dict(temperature=temperature), stream=True)

st.set_page_config(layout="wide")    
st.title('LLM Arena')
st.markdown('##### Evaluate your prompt against all your local LLMs.')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

def streaming_callback(x):
        for chunk in x:
            yield chunk["response"]

if 'prompt' not in st.session_state:
    if prompt := st.chat_input('Your prompt'):
        with st.chat_message('user'):
            st.write(prompt)
            st.session_state.messages.append(dict(role='user', content=prompt))

        for LLM in LLMs:
            with st.chat_message('ai'):
                response = llm(prompt, LLM)
                prefix = '**' + LLM + '**'
                st.write(prefix)
                answer = st.write_stream(streaming_callback(response))
                st.session_state.messages.append(dict(role='ai', content=prefix + '\n' + answer))
        
        st.session_state.prompt = prompt
        st.rerun()

elif 'eval' not in st.session_state and st.button('Evaluate', type='primary'):
    st.markdown('**Judge:** _%s_' % JUDGE)
    prompt = EVAL_PROMPT % (st.session_state.prompt, '\n\n\n'.join([x['content'] for x in st.session_state.messages[1:]]))
    if JUDGE == 'chatgpt':
        st.text_area('Judgement prompt', prompt)
    else:
        with st.chat_message('user'):
                st.write(prompt)
                st.session_state.messages.append(dict(role='user', content=prompt))
        with st.chat_message('ai'):
            response = llm(prompt, JUDGE)
            answer = st.write_stream(streaming_callback(response))
            st.session_state.messages.append(dict(role='ai', content=answer))
            st.session_state.eval = answer
else:
    st.stop()
