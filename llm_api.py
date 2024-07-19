import os
from openai import OpenAI

API_KEY = os.getenv("OPENAI_API_KEY") # Your API_KEY Here

def run_gpt_chat(messages):
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://api.openai-proxy.com/v1"
    )
    attempt = 0
    while attempt < 3:
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                temperature=0,
                top_p=0,
                messages=messages,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print("GPT Generate Error:", e)
            attempt += 1
    return "Error"

def run_llm_chat(messages):
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1"
    )
    attempt = 0
    while attempt < 3:
        try:
            completion = client.chat.completions.create(
                model="Meta-Llama-3-8B-Instruct",
                temperature=0,
                top_p=0,
                messages=messages,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print("LLM Generate Error:", e)
            attempt += 1
    return "Error"

def messages_to_prompt(messages):
    return '\n'.join([message['content'] for message in messages])