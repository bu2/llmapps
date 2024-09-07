import ollama
import streamlit as st

LLMs = [
    'qwen2-math:1.5b',
    'yi-coder:1.5b',
    'gemma2:2b',
    
    # 'phi:latest',
    
    # 'phi3:latest',
    'phi3.5:latest',
    
    # 'llama2:latest',
    
    'mathstral:latest',
    'mistral:latest',
    'qwen2:latest',
    'qwen2-math:latest',
    
    # 'internlm2:latest',
    
    # 'llama3:latest',
    'llama3.1:latest',
    'hermes3:latest',
    'aya:latest',
    'yi-coder:latest',

    # 'gemma:latest',
    'gemma2:latest',

    # 'glm4:latest',
    # 'codegeex4:latest',
    
    'mistral-nemo:latest',
    'phi3:14b',
    'deepseek-v2:latest',
    'deepseek-coder-v2:latest',
    
    # 'starcoder2:15b',

    'codestral:latest',
    'gemma2:27b',
    
    'deepseek-coder:33b',

    # 'granite-code:34b',
    
    'yi:34b',
    'aya:35b',
    'command-r:latest',
    'mixtral:latest',
    'llama3:70b',
    'llama3.1:70b',
    'hermes3:70b',
    'reflection:latest',
    'qwen2:72b',
    'qwen2-math:72b',
    'command-r-plus:latest',
    'qwen:110b',
    'mistral-large:latest',
    
    # 'dbrx:latest',
    
    'mixtral:8x22b',
    'wizardlm2:8x22b',
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
