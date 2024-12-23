import json
from json.decoder import JSONDecodeError
import re
import time

import numpy as np
import ollama
import pandas as pd
import streamlit as st
from tqdm import tqdm

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
chromadb.api.client.SharedSystemClient.clear_system_cache()


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
    'granite3.1-dense',
    'gemma2',
    'falcon3:10b',
    'mistral-nemo',
    'phi3:14b',
    'qwen2.5:14b',
    'mistral-small',
    'codestral',
    'gemma2:27b',
    'command-r',
    'qwen2.5:32b',
    'qwq',
    'mixtral',
    'llama3.1:70b',
    'llama3.3:70b',
    'hermes3:70b',
    'tulu3:70b',
    'qwen2.5:72b',
    'command-r-plus',
    'qwen:110b',
    'mistral-large',
    'mixtral:8x22b',
]

DATASET = 'data/arxiv/arxiv-metadata-oai-snapshot.json'
LLM = 'mistral-large'
EMBEDDING = 'snowflake-arctic-embed2'
K = 10
TEMPERATURE = 0
LIMIT = 1000

COLS = ['id', 'update_date', 'submitter', 'authors', 'title', 'categories', 'license', 'abstract']

SYSTEM_KEYWORDS = '''
Summarize the following prompt to a very short list of specific keywords to search in research papers:
%s


Use lowercase, avoid plural and keep any compound words as single keyword with spaces.
Output the keywords as a valid JSON list of strings.
'''


@st.cache_resource
def print_banner():
    print(r'''
          
    _       __  ___             __  __      _                     
   / \   _ _\ \/ (_)_   __   ___\ \/ /_ __ | | ___  _ __ ___ _ __ 
  / _ \ | '__\  /| \ \ / /  / _ \\  /| '_ \| |/ _ \| '__/ _ \ '__|
 / ___ \| |  /  \| |\ V /  |  __//  \| |_) | | (_) | | |  __/ |   
/_/   \_\_| /_/\_\_| \_/    \___/_/\_\ .__/|_|\___/|_|  \___|_|   
                                     |_|                          


''')


@st.cache_resource
def load_arxiv_dataset():
    print('Load ArXiv dataset...')
    tstart = time.perf_counter()
    df = pd.read_json(DATASET, orient='records', lines=True, convert_dates=['update_date']).sort_values('update_date', ascending=False)[COLS]
    tend = time.perf_counter()
    print('%fs elapsed.' % (tend-tstart))
    print('%d documents...' % len(df))
    return df


@st.cache_resource
def filter_arxiv_dataset(df, keywords):
    print('Filter documents...')
    tstart = time.perf_counter()
    if len(keywords) > 0:
        mask = None
        scores = None
        for kw in keywords:
            matches = (df['abstract'].str.upper().str.find(kw.upper()) > -1)
            if mask is None:
                mask = matches
                scores = matches.astype(int)
            else:
                mask = mask | matches
                scores = scores + matches.astype(int)
        df['score'] = scores
        df = df[mask]
    df = df.sort_values(['score', 'update_date'], ascending=False) \
           .iloc[:LIMIT]                                           \
           .reset_index(drop=True)
    texts = []
    for record in df.to_dict('records'):
        texts.append("Date: %s\nAuthors: %s\nTitle: %s\nAbstract: %s\n" % (record['title'], record['update_date'], record['authors'], record['abstract']))
    df['text'] = texts
    tend = time.perf_counter()
    print('%fs elapsed.' % (tend-tstart))
    print('%d filtered documents...' % len(df))
    return df


@st.cache_resource
def compute_embeddings(df):
    print('Compute embeddings...')
    tstart = time.perf_counter()
    embedding = OllamaEmbeddingFunction(url="http://localhost:11434/api/embeddings", 
                                        model_name=EMBEDDING)
    df['vectors'] = list(embedding([str(text) for text in df['text'].values]))
    tend = time.perf_counter()
    print('%fs elapsed.' % (tend-tstart))
    print(df.info())
    return df


@st.cache_resource
def build_index(df):
    print('Build index...')
    tstart = time.perf_counter()
    db = chromadb.Client()
    embedding = OllamaEmbeddingFunction(url="http://localhost:11434/api/embeddings", 
                                        model_name=EMBEDDING)
    docs = db.create_collection('docs', embedding_function=embedding)
    for i, item in enumerate(tqdm(df.to_dict(orient='records'))):
        docs.add(
            documents=item['text'], 
            ids=str(i), 
            embeddings=item['vectors'], 
            metadatas={k:(v or '') for k, v in item.items() if k not in ('text', 'vectors', 'update_date')}
        )
    tend = time.perf_counter()
    print('%fs elapsed.' % (tend-tstart))
    return docs


def llm(prompt, stream=True):
    return ollama.generate(prompt=prompt, model=LLM, stream=stream)


def streaming_callback(x):
    for chunk in x:
        yield chunk['response']


st.set_page_config(layout="wide")    
print_banner()
st.title('ArXiv eXplorer')
st.markdown('##### Leverage LLMs with RAG to explore the [ArXiv abstracts dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv).')

df = load_arxiv_dataset()

LLM = st.selectbox('LLM:', LLMs, LLMs.index(LLM))


st.session_state.history = st.session_state.get('history', [])
for item in st.session_state.history:
    with st.chat_message(item['role']):
        if type(item['message']) is str:
            st.markdown(item['message'])
        elif type(item['message']) is pd.DataFrame:
            st.dataframe(item['message'])


if prompt := st.chat_input('Ask a scientific question and get answers from the ArXiv...'):
    with st.chat_message('user'):
        st.markdown(prompt)
    st.session_state.history.append(dict(role='user', message=prompt))
    answer = llm(SYSTEM_KEYWORDS % prompt, stream=False)['response']
    try:
        json.loads(answer)
    except JSONDecodeError:
        print('[KEYWORDS]')
        print(answer)
        print('[/KEYWORDS]')
        m = re.match(r'.*```(json)?\n(.*)\n```.*', answer, flags=re.MULTILINE|re.DOTALL)
        answer = m.groups()[1]
    keywords = json.loads(answer)
    keywords_str = '**Key words:** __' + repr(keywords) + '__'
    st.markdown(keywords_str)
    st.session_state.history.append(dict(role='assistant', message=keywords_str))
    df = filter_arxiv_dataset(df, keywords)
    df = compute_embeddings(df)
    docs = build_index(df)
    rez = docs.query(query_texts=prompt, n_results=K)

    rezdf = df.iloc[rez['ids'][0]]
    rezdf['distance'] = rez['distances'][0]
    rezdf.sort_values('update_date', inplace=True)
    rezdf.reset_index(drop=True, inplace=True)
    st.dataframe(rezdf[['distance', 'score', 'id', 'update_date', 'authors', 'title']])
    st.session_state.history.append(dict(role='assistant', message=rezdf[['distance', 'score', 'id', 'update_date', 'authors', 'title']]))

    ctx = rezdf['text'].values
    response = llm(f'''
<CONTEXT> 
{'\n\n'.join(ctx)}
</CONTEXT>

Prompt: {prompt}

Analyse all the information from the context in your mind and use it to produce a detailed step-by-step answer to the prompt.
Justify your answer with examples and references from the context.
Add a table of unique references (date, authors, title) from the context after your answer.
''')
    answer = st.write_stream(streaming_callback(response))
    st.session_state.history.append(dict(role='assistant', message=answer))
    st.rerun()
