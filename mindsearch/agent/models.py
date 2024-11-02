import os
from dotenv import load_dotenv
from lagent.llms import GPTAPI

load_dotenv()

gpt4 = dict(
    type=GPTAPI,
    model_type=os.environ.get('OPENROUTER_MODEL', 'gpt-4o-mini'),
    key=os.environ.get('OPENROUTER_API_KEY', 'YOUR OPENROUTER API KEY'),
    openai_api_base='https://openrouter.ai/api/v1/chat/completions',
)

ollama = dict(
    type=GPTAPI,
    model_type=os.environ.get('OLLAMA_MODEL', 'mindsearch:latest'),
    key=None,
    openai_api_base=os.environ.get('OLLAMA_API_BASE', 'http://localhost:11434/v1/chat/completions'),
)
