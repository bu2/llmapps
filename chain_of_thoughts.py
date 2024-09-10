from datetime import datetime
import json
from json.decoder import JSONDecodeError
import os
import re

import ollama
import streamlit as st
import tiktoken


LLMs = [
    'gemma2:2b',
    'phi3.5',
    'mistral',
    'qwen2',
    'llama3.1',
    'gemma2',
    'command-r',
    'mistral-nemo',
    'phi3:14b',
    'codestral',
    'gemma2:27b',
    'mixtral',
    'llama3.1:70b',
    'reflection',
    'qwen2:72b',
    'command-r-plus',
    'qwen:110b',
    'mistral-large',
    'mixtral:8x22b',
]

DEFAULT_INSTRUCTOR = 'mistral-large'
DEFAULT_EXECUTOR = 'codestral'

SYSTEM_COT = 'Break down the complex problem from the prompt and derive a plan to solve the problem. Write only the plan at this stage.\n'

SYSTEM_PLAN = 'Extract the plan, improve it and sort the tasks in the best order to avoid duplicate work. **Output only the optimized plan as a JSON list of strings!**\n'

SYSTEM_TASK = 'Process the task based on the context without repeating what is already in the context.\n'

SYSTEM_PROMPT = 'Answer the prompt based on the context.'

STATE_FILE = 'cot_session_state.json'


tokenizer = tiktoken.encoding_for_model('gpt-4o')

    
def llm(prompt, llm_):
    return ollama.generate(prompt=prompt, model=llm_, options={'temperature': 0}, stream=True)


def streaming_callback(x):
    for chunk in x:
        yield chunk['response']


st.set_page_config(layout="wide")    
st.title('Chain of Thoughts')
st.markdown('##### Ask a complex problem and follow the chain of thoughts...')


if os.path.exists(STATE_FILE):
    with open(STATE_FILE) as f:
        state = json.load(f)
    INSTRUCTOR = state['instructor']
    EXECUTOR = state['executor']
else:
    state = {'instructor': DEFAULT_INSTRUCTOR, 'executor': DEFAULT_EXECUTOR, 'prompt': None, 'history': []}


st.session_state.prompt = st.session_state.get('prompt', state['prompt'])
st.session_state.history = st.session_state.get('history', state['history'])


INSTRUCTOR = st.selectbox('Instructor:', LLMs, LLMs.index(DEFAULT_INSTRUCTOR))
EXECUTOR = st.selectbox('Executor:', LLMs, LLMs.index(DEFAULT_EXECUTOR))


if st.button('Reset', type='primary', disabled=not os.path.exists(STATE_FILE)):
    os.unlink(STATE_FILE)
    state = {'instructor': DEFAULT_INSTRUCTOR, 'executor': DEFAULT_EXECUTOR, 'prompt': None, 'history': []}
    INSTRUCTOR = DEFAULT_INSTRUCTOR
    EXECUTOR = DEFAULT_EXECUTOR
    st.session_state.clear()
    st.rerun()


if len(st.session_state.history) > 0:
    for item in st.session_state.history:
        with st.chat_message(item['role']):
            st.markdown(item['message'])


if st.session_state.prompt is None:
    if prompt := st.chat_input('Submit a complex problem...'):
        st.session_state.prompt = prompt
        st.rerun()

elif len(st.session_state.history) == 0:

    st.chat_input('Busy...', disabled=True)

    tstart = datetime.now()
    
    prompt = st.session_state.prompt
    with st.chat_message('user'):
        st.markdown(prompt)
    st.session_state.history.append({'role': 'user', 'message': prompt})
    response = llm('## Prompt:\n' + prompt + '\n\n\n' + SYSTEM_COT, INSTRUCTOR)
    with st.chat_message('ai'):
        answer = st.write_stream(streaming_callback(response))
    st.session_state.history.append({'role': 'ai', 'message': answer})
    response = llm(answer + '\n\n\n' + SYSTEM_PLAN, INSTRUCTOR)
    with st.chat_message('assistant'):
        st.markdown('### Action plan')
        answer = st.write_stream(streaming_callback(response))
    answer = answer.strip()
    st.session_state.history.append({'role': 'assistant', 'message': '### Action plan:\n' + answer})

    try:
        json.loads(answer)
    except JSONDecodeError:
        print('[PLAN]')
        print(answer)
        print('[/PLAN]')
        m = re.match(r'.*```(json)?\n(.*)\n```.*', answer, flags=re.MULTILINE|re.DOTALL)
        answer = m.groups()[1]

    steps = json.loads(answer)
    for i, step_ in enumerate(steps):
        step = '[%d/%d] %s' % (i+1, len(steps), step_)
        with st.chat_message('assistant'):
            st.markdown('### ' + step)
        
        prompt2 = 'Context:\n\n'
        for item in st.session_state.history:
            prompt2 += f"\t{item['role']}: {item['message']}\n\n"
        prompt2 += f"\n\n\n## Task\n{step}"
        prompt2 += '\n\n\n' + SYSTEM_TASK 
        response = llm(prompt2, EXECUTOR)
        with st.chat_message('ai'):
            answer = st.write_stream(streaming_callback(response))
        st.session_state.history.append({'role': 'assistant', 'message': '### ' + step})
        st.session_state.history.append({'role': 'ai', 'message': answer})

    tend = datetime.now()
    cot_time = tend - tstart

    blob = '\n\n'.join([x['message'] for x in st.session_state.history])
    tokens = tokenizer.encode(blob)
    st.session_state.history.append({'role': 'assistant', 'message': '**CoT time: %s, Context size: %d tokens**' % (str(cot_time), len(tokens))})

    with open(STATE_FILE, 'w') as f:
        state = {
            'instructor': INSTRUCTOR,
            'executor': EXECUTOR,
            'prompt': st.session_state.prompt,
            'history': st.session_state.history
        }
        json.dump(state, f)
            
    st.rerun()

else:
    if prompt := st.chat_input('Any remaining question ?'):
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.history.append({'role': 'user', 'message': prompt})
        
        prompt2 = 'Context:\n\n'
        for item in st.session_state.history:
            prompt2 += f"\t{item['role']}: {item['message']}\n\n"
        prompt2 += f"\n\n\n## Task\n{prompt}"
        prompt2 += '\n\n\n' + SYSTEM_PROMPT
        response = llm(prompt2, INSTRUCTOR)
        with st.chat_message('ai'):
            answer = st.write_stream(streaming_callback(response))
            st.session_state.history.append({'role': 'ai', 'message': answer})

        blob = '\n\n'.join([x['message'] for x in st.session_state.history])
        tokens = tokenizer.encode(blob)
        st.session_state.history.append({'role': 'assistant', 'message': '**Context size: %d tokens**' % len(tokens)})

        with open(STATE_FILE, 'w') as f:
            state = {
                'instructor': INSTRUCTOR,
                'executor': EXECUTOR,
                'prompt': st.session_state.prompt,
                'history': st.session_state.history
            }
            json.dump(state, f)

        st.rerun()
