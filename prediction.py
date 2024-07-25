import os
import argparse
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from data_manager import DataManager
from prompt_templates import SELECTING_PROMPT, REASONING_PROMPT, REASONING_LONGTAIL_PROMPT

from llm_api import run_llm_chat
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["FB15k-237-subset", "NELL-995-subset", "WN18RR-subset"], default="FB15k-237-subset", help="Name of the dataset")
    parser.add_argument("--setting", type=str, choices=["inductive", "transductive"], default="inductive", help="Inductive or Transductive setting")
    parser.add_argument("--train_size", type=str, choices=["full", "1000", "2000"], default="full", help="Size of the training data")
    
    args = parser.parse_args()

    data_manager = DataManager(dataset=args.dataset, setting=args.setting, train_size=args.train_size)
    test_batches = data_manager.get_test_batches()[:100]
    
    model_path = "/data/FinAi_Mapping_Knowledge/finllm/LLMs/Qwen2-7B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto",device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    generation_config = dict(
        temperature=0,
        top_k=0,
        top_p=0,
        do_sample=False,
        max_new_tokens=1,
    )

    device = "cuda"
    Y_id = tokenizer.encode("Y", add_special_tokens=False)[0]
    N_id = tokenizer.encode("N", add_special_tokens=False)[0]
    
    select_result = []
    reason_result = []
    score_result = []
    prob_result = []
    hits_result = []
    for idx, batch in enumerate(tqdm(test_batches, desc="Processing batches")):
        select_in_batch = []
        reason_in_batch = []
        score_in_batch = []
        prob_in_batch = []
        for batch_idx, test_triple in enumerate(batch):
            test_head, test_relation, test_tail = test_triple

            fewshot_triples = data_manager.diverse_fewshot_triple_finder(test_triple, 5)
            test_triple_sentence = data_manager.triple_to_sentence(test_triple)
            fewshot_triples_sentence = '\n'.join(data_manager.triple_to_sentence(triple) for triple in fewshot_triples)
            selecting_prompt = SELECTING_PROMPT.format(fewshot_triples=fewshot_triples_sentence, test_triple=test_triple_sentence)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": selecting_prompt}
            ]
            # print(selecting_prompt)
            # select_response = run_llm_chat([{"role": "user", "content": selecting_prompt}]).strip()
            # print(select_response)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(device)

            generated_output = model.generate(
                input_ids=inputs.input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                **generation_config
            )

            scores = generated_output.scores[0]
            probs = scores.softmax(dim=-1)
            Y_score = scores[0, Y_id].item()
            N_score = scores[0, N_id].item()
            score_in_batch.append(Y_score)
            Y_prob = probs[0, Y_id].item()
            N_prob = probs[0, N_id].item()
            prob_in_batch.append(Y_prob)
        top_indices = sorted(range(len(prob_in_batch)), key=lambda i: prob_in_batch[i], reverse=True)[:10]
        if 0 in top_indices:
            select_result.append(1)
        else:
            select_result.append(0)
        top_prob_with_indices = [(idx, prob_in_batch[idx]) for idx in top_indices]
        reasoning_probs = []
        for idx in top_indices:
            test_triple = batch[idx]
            test_head, test_relation, test_tail = test_triple
            test_head_tail = f"{test_head}-{test_tail}"
            test_close_paths = data_manager.close_path_dict[test_head_tail]
            if test_close_paths != []:
                reasoning_paths = "\n".join(
                    " -> ".join(data_manager.triple_to_sentence(triple) for triple in path)
                    for path in test_close_paths
                )
                reasoning_prompt = REASONING_PROMPT.format(reasoning_paths=reasoning_paths, test_triple=data_manager.triple_to_sentence(test_triple))
            else:
                known_triples = data_manager.related_triple_finder(test_triple, 3)
                reasoning_prompt = REASONING_LONGTAIL_PROMPT.format(known_triples='\n'.join(known_triples), test_triple=data_manager.triple_to_sentence(test_triple))
            # print(reasoning_prompt)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": reasoning_prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(reasoning_prompt, return_tensors="pt").to(device)
            generated_output = model.generate(
                input_ids=inputs.input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                **generation_config
            )
            probs = generated_output.scores[0].softmax(dim=-1)
            Y_prob = probs[0, Y_id].item()
            reasoning_probs.append(Y_prob)
        sorted_reasoning_indices = sorted(range(len(reasoning_probs)), key=lambda i: reasoning_probs[i], reverse=True)
        hits_position = sorted_reasoning_indices.index(top_indices.index(0)) + 1 if 0 in top_indices else 0
        hits_result.append(hits_position)

        # select_result.append(response_in_batch)
        # reason_result.append(reason_in_batch)
        # score_result.append(score_in_batch)
        # prob_result.append(prob_in_batch)
    print("Recall rate:", sum(select_result) / len(select_result))
        
    print("Hits results:", hits_result)
    print("Hit@1:", sum(1 for hits in hits_result if hits == 1) / len(hits_result))
    print("MRR:", sum(1 / hits for hits in hits_result if hits != 0) / len(hits_result))
    

if __name__ == "__main__":
    main()
