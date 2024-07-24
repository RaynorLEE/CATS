import os
import argparse
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    test_batches = data_manager.get_test_batches()[:20]
    
    # tokenizer = AutoTokenizer.from_pretrained("/data/FinAi_Mapping_Knowledge/LLMs/Meta-Llama-3-8B-Instruct")
    # model = AutoModelForCausalLM.from_pretrained("/data/FinAi_Mapping_Knowledge/LLMs/Meta-Llama-3-8B-Instruct")
    
    select_result = []
    reason_result = []
    for idx, batch in enumerate(tqdm(test_batches, desc="Processing batches")):
        response_record = []
        reason_in_batch = []
        for batch_idx, test_triple in enumerate(batch):
            test_head, test_relation, test_tail = test_triple

            fewshot_triples = data_manager.diverse_fewshot_triple_finder(test_triple, 5)
            test_triple_sentence = data_manager.triple_to_sentence(test_triple)
            fewshot_triples_sentence = '\n'.join(data_manager.triple_to_sentence(triple) for triple in fewshot_triples)
            test_head_name = f"'{data_manager.entity2text[test_head]}'"
            test_tail_name = f"'{data_manager.entity2text[test_tail]}'"
            fewshot_heads = '(' + ', '.join(f"'{data_manager.entity2text[head]}'" for head, _, _ in fewshot_triples) + ')'
            fewshot_tails = '(' + ', '.join(f"'{data_manager.entity2text[tail]}'" for _, _, tail in fewshot_triples) + ')'
            selecting_prompt = SELECTING_PROMPT.format(fewshot_triples=fewshot_triples_sentence, test_triple=test_triple_sentence, test_head=test_head_name, test_tail=test_tail_name, fewshot_heads=fewshot_heads, fewshot_tails=fewshot_tails)
            # print(selecting_prompt)
            select_response = run_llm_chat([{"role": "user", "content": selecting_prompt}]).strip()
            # print(select_response)
            if select_response == 'Y':
                response_record.append(1)
            else:
                response_record.append(0)
            
            # if select_response == 'Y':
            #     test_head_tail = f"{test_head}-{test_tail}"
            #     test_close_paths = data_manager.close_path_dict[test_head_tail]
            #     reasoning_paths = "\n".join(
            #         " -> ".join(data_manager.triple_to_sentence(triple) for triple in path)
            #         for path in test_close_paths
            #     )
                
            #     # 不考虑longtail，没有close path就算了
            #     reasoning_prompt = REASONING_PROMPT.format(reasoning_paths=reasoning_paths, test_triple=data_manager.triple_to_sentence(test_triple))
                
            #     # 考虑longtail，没有close path，就用related triple作为补充信息
            #     # if test_close_paths != []:
            #     #     reasoning_paths = "\n".join(
            #     #         " -> ".join(data_manager.triple_to_sentence(triple) for triple in path)
            #     #         for path in test_close_paths
            #     #     )
            #     #     reasoning_prompt = REASONING_PROMPT.format(reasoning_paths=reasoning_paths, test_triple=data_manager.triple_to_sentence(test_triple))
            #     # else:
            #     #     known_triples = data_manager.related_triple_finder(test_triple, 3)
            #     #     reasoning_prompt = REASONING_LONGTAIL_PROMPT.format(known_triples='\n'.join(known_triples), test_triple=data_manager.triple_to_sentence(test_triple))
                
            #     # print(reasoning_prompt)
            #     reasoning_response = run_llm_chat([{"role": "user", "content": reasoning_prompt}])
            #     # print(reasoning_response)
                
            #     if reasoning_response == 'Y':
            #         reason_in_batch.append(batch_idx + 1)
        
        reason_result.append(reason_in_batch)
        select_result.append(response_record)
    
    num_batches = len(select_result)
    count_first_ones = sum(1 for record in select_result if record[0] == 1)
    proportion_first_ones = count_first_ones / num_batches
    ones_per_batch = [sum(record) for record in select_result]
    total_ones = sum(ones_per_batch)
    total_elements = sum(len(record) for record in select_result)
    proportion_ones = total_ones / total_elements

    print(f"Proportion of batches where the first element is 1: {proportion_first_ones:.2f}")
    print(f"Number of ones in each batch: {ones_per_batch}")
    print(f"Proportion of ones across all batches: {proportion_ones:.2f}")

    print("Reasoning results:", reason_result)
    for result in select_result:
        print(result)

if __name__ == "__main__":
    main()
