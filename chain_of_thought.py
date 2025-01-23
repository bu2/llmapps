from datetime import datetime
import json
from json.decoder import JSONDecodeError
import os
import re

import ollama
import streamlit as st
import tiktoken


LLMs = [
    'llama3.2:3b',
    'qwen2.5:3b',
    'falcon3:3b',
    'phi3.5',
    'mistral',
    'qwen2.5',
    'qwen2.5-coder',
    'llama3.1',
    'hermes3',
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
    'deepseek-r1:14b',
    'mistral-small',
    'codestral',
    'gemma2:27b',
    'command-r',
    'qwen2.5:32b',
    'qwq',
    'deepseek-r1:32b',
    'mixtral',
    'llama3.1:70b',
    'llama3.3:70b',
    'hermes3:70b',
    'tulu3:70b',
    'deepseek-r1:70b',
    'qwen2.5:72b',
    'command-r-plus',
    'qwen:110b',
    'mistral-large',
    'mixtral:8x22b',
]

DEFAULT_INSTRUCTOR = 'deepseek-r1:70b'
DEFAULT_EXECUTOR = DEFAULT_INSTRUCTOR

SYSTEM_COT = 'Think about how to crack the complex problem and define a plan to solve it. Output only the plan in plain text, as a flat list of tasks.'

SYSTEM_PLAN = 'Extract all tasks with their full description and output them as a JSON list of strings.\n'

SYSTEM_TASK = 'Use the context and process the task to progress toward the solution to the user problem.\n'

SYSTEM_PROMPT = 'Use the context to answer the prompt.\n'

STATE_FILE = 'cot_session_state.json'


tokenizer = tiktoken.encoding_for_model('gpt-4o')

    
def llm(prompt, llm_):
    return ollama.generate(prompt=prompt, model=llm_, options={'temperature': 0}, stream=True)


def streaming_callback(x):
    for chunk in x:
        yield chunk['response']


st.set_page_config(layout="wide")    
st.title('Chain of Thought')
st.markdown('##### Ask a complex problem and follow the chain of thought...')


if os.path.exists(STATE_FILE):
    with open(STATE_FILE) as f:
        state = json.load(f)
    INSTRUCTOR = state['instructor']
    EXECUTOR = state['executor']
else:
    state = {
        'instructor': DEFAULT_INSTRUCTOR,
        'executor': DEFAULT_EXECUTOR,
        'do_planning': True,
        'prompt': None,
        'plan': None,
        'go': False,
        'history': []
    }


st.session_state.prompt = st.session_state.get('prompt', state['prompt'])
st.session_state.do_planning = st.session_state.get('do_planning', state['do_planning'])
st.session_state.plan = st.session_state.get('plan', state['plan'])
st.session_state.go = st.session_state.get('go', state['go'])
st.session_state.history = st.session_state.get('history', state['history'])


INSTRUCTOR = st.selectbox('Instructor:', LLMs, LLMs.index(DEFAULT_INSTRUCTOR))
EXECUTOR = st.selectbox('Executor:', LLMs, LLMs.index(DEFAULT_EXECUTOR))

st.session_state.do_planning = st.checkbox('Plan', value=st.session_state.do_planning)


if st.button('Reset', type='primary', disabled=not os.path.exists(STATE_FILE)):
    if os.path.exists(STATE_FILE):
        os.unlink(STATE_FILE)
    state = {
        'instructor': DEFAULT_INSTRUCTOR,
        'executor': DEFAULT_EXECUTOR,
        'do_planning': True,
        'prompt': None,
        'plan': None,
        'go': False,
        'history': []
    }
    INSTRUCTOR = DEFAULT_INSTRUCTOR
    EXECUTOR = DEFAULT_EXECUTOR
    st.session_state.clear()
    st.rerun()


if len(st.session_state.history) > 0:
    for item in st.session_state.history:
        with st.chat_message(item['role']):
            st.markdown(item['message'])


def build_prompt(problem=None):
    if problem:
        prompt = f"## User problem:\n{problem}\n\n\n"
    else:
        prompt = ''
    prompt += '## Context:\n\n'
    for item in st.session_state.history:
        prompt += f"\t{item['role']}: {item['message']}\n\n"
    prompt += '\n'
    return prompt


if st.session_state.prompt is None:
    if prompt := st.chat_input('Submit a complex problem...'):
        st.session_state.prompt = prompt
        st.rerun()

elif st.session_state.do_planning and st.session_state.plan is None:
    st.chat_input('Busy...', disabled=True)
    
    prompt = st.session_state.prompt
    with st.chat_message('user'):
        st.markdown(prompt)

    prompt2 = build_prompt()
    prompt2 += '## Problem:\n' + prompt + '\n\n\n' + SYSTEM_COT
    response = llm(prompt2, INSTRUCTOR)
    with st.chat_message('ai'):
        answer = st.write_stream(streaming_callback(response))
    st.session_state.history.append({'role': 'user', 'message': prompt})
    st.session_state.plan = answer
    st.session_state.go = True

    st.text_area('Plan', key='plan')
    st.button('Go', type='primary')

elif st.session_state.do_planning and st.session_state.go:
    st.session_state.history.append({'role': 'ai', 'message': st.session_state.plan})

    tstart = datetime.now()
    
    response = llm(st.session_state.plan + '\n\n\n' + SYSTEM_PLAN, INSTRUCTOR)
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
        
        prompt2 = build_prompt(st.session_state.prompt)
        prompt2 += f"\n\n\n## Task:\n{step}"
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

    st.session_state.do_planning = False
    st.session_state.prompt = None
    st.session_state.plan = None
    st.session_state.go = False

    with open(STATE_FILE, 'w') as f:
        state = {
            'instructor': INSTRUCTOR,
            'executor': EXECUTOR,
            'do_planning': False,
            'prompt': None,
            'plan': None,
            'go': False,
            'history': st.session_state.history
        }
        json.dump(state, f)
            
    st.rerun()

else:
    st.chat_input('Busy...', disabled=True)
    
    prompt = st.session_state.prompt
    with st.chat_message('user'):
        st.markdown(prompt)
    st.session_state.history.append({'role': 'user', 'message': prompt})
    
    prompt2 = build_prompt()
    prompt2 += f"\n\n\n## Prompt:\n{prompt}"
    prompt2 += '\n\n\n' + SYSTEM_PROMPT
    response = llm(prompt2, INSTRUCTOR)
    with st.chat_message('ai'):
        answer = st.write_stream(streaming_callback(response))
        st.session_state.history.append({'role': 'ai', 'message': answer})

    blob = '\n\n'.join([x['message'] for x in st.session_state.history])
    tokens = tokenizer.encode(blob)
    st.session_state.history.append({'role': 'assistant', 'message': '**Context size: %d tokens**' % len(tokens)})

    st.session_state.prompt = None

    with open(STATE_FILE, 'w') as f:
        state = {
            'instructor': INSTRUCTOR,
            'executor': EXECUTOR,
            'do_planning': False,
            'prompt': None,
            'plan': None,
            'go': False,
            'history': st.session_state.history
        }
        json.dump(state, f)

    st.rerun()
