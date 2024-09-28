# LLM Apps

LLM experiments based on Ollama and Streamlit.

### Prerequisites
You need a local install of Ollama (https://ollama.com) and the following Python packages:
```bash
$ pip install chroma ollama pandas streamlit sentence-transformers tiktoken
```

### Code Assistant
Use your favourite local LLM as code assistant to generate code and iterate on it.
```bash
$ streamlit run code_assistant.py
```

### LLM Arena
Evaluate your prompt against all your local LLMs.
```bash
$ streamlit run llm_arena.py
```

### Chain of Thought
Leverage the Chain-of-Thought principle to crack complex problems.
```bash
$ streamlit run chain_of_thought.py
```
