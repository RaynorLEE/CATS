import os
import argparse
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from data_manager import DataManager
from prompt_templates import EXPLAINING_PROMPT

from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # model_path = "/data/FinAi_Mapping_Knowledge/LLMs/Qwen2-7B-Instruct"
    model_path = "/data/FinAi_Mapping_Knowledge/LLMs/Qwen2-7B-Instruct-fb-v0"
    # model_path = "/data/FinAi_Mapping_Knowledge/LLMs/Qwen1.5-14B-Chat"

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto",device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    generation_config = dict(
        temperature=0,
        top_k=0,
        top_p=0,
        do_sample=False,
        max_new_tokens=200,
    )
    device = "cuda"
    known_triples = ""
    reasoning_paths = ""
    test_triple = ("entity_1", "relation", "entity_2")
    explain_prompt = EXPLAINING_PROMPT.format(known_triples=known_triples, reasoning_paths=reasoning_paths, test_triple=test_triple)
    explain_messages = [
        {"role": "user", "content": explain_prompt}
    ]
    explain_text = tokenizer.apply_chat_template(
        explain_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(explain_prompt, return_tensors="pt").to(device)

    generated_output = model.generate(
        input_ids=inputs.input_ids,
        **generation_config
    )
    print(tokenizer.decode(generated_output[0], skip_special_tokens=True))