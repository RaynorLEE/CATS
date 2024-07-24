import os
import json
import random
from collections import defaultdict, deque
from data_manager import DataManager
from tqdm import tqdm
import argparse
from prompt_templates import SELECTING_PROMPT, REASONING_PROMPT, REASONING_LONGTAIL_PROMPT
from llm_api import run_llm_chat

def bfs_paths(entity2relationtail_dict, start, goal, max_hops, max_paths):
    queue = deque([(start, [], 0, set([start]))])
    paths = []
    while queue:
        current, path, hops, visited = queue.popleft()
        if hops < max_hops:
            for relation, neighbor, direction in entity2relationtail_dict[current]:
                if direction == 1:
                    new_path = path + [(current, relation, neighbor)]
                else:
                    new_path = path + [(neighbor, relation, current)]
                if neighbor == goal:
                    paths.append(new_path)
                elif neighbor not in visited:
                    queue.append((neighbor, new_path, hops + 1, visited | set([neighbor])))
    return paths[:max_paths]

def high_quality_neg_sampling(data_manager: DataManager, neg_samples):
    high_quality_neg_samples = []
    for neg_triple in neg_samples:
        fewshot_triples = data_manager.diverse_fewshot_triple_finder(neg_triple, 10)
        selecting_prompt = SELECTING_PROMPT.format(fewshot_triples="\n".join(data_manager.triple_to_sentence(triple) for triple in fewshot_triples), test_triple=data_manager.triple_to_sentence(neg_triple))
        selecting_response = run_llm_chat([{"role": "user", "content": selecting_prompt}])
        if selecting_response == 'Y':
            high_quality_neg_samples.append(neg_triple)
            
    return high_quality_neg_samples

def build_instructions(dataset, max_path_hop, max_paths, train_size):
    setting = "transductive" # 指令构建默认是transductive，用训练集
    
    data_manager = DataManager(dataset=dataset, setting=setting, train_size=train_size)

    paths_dir = f"instructions/{dataset}"
    os.makedirs(paths_dir, exist_ok=True)

    close_path_dict = {}
    close_paths_text = []
    sft_instructions = []
    
    for pos_triple in tqdm(data_manager.train_set[:5], desc=f"Processing {dataset} - setting: {setting} - Train_size: {train_size}"):
        pos_head, relation, pos_tail = pos_triple
        removed_from_head = (relation, pos_tail, 1)
        removed_from_tail = (relation, pos_head, -1)
        
        # 移除当前triple，因为是寻找当前(pos_head, pos_tail)的close_path
        if removed_from_head in data_manager.entity2relationtail_dict[pos_head]:
            data_manager.entity2relationtail_dict[pos_head].remove(removed_from_head)
        if removed_from_tail in data_manager.entity2relationtail_dict[pos_tail]:
            data_manager.entity2relationtail_dict[pos_tail].remove(removed_from_tail)

        pos_paths = list(bfs_paths(data_manager.entity2relationtail_dict, pos_head, pos_tail, max_path_hop, max_paths))
        close_path_dict[f"{pos_head}-{pos_tail}"] = pos_paths
        
        data_manager.entity2relationtail_dict[pos_head].append(removed_from_head)
        data_manager.entity2relationtail_dict[pos_tail].append(removed_from_tail)
        
        for path_pair in pos_paths:
            path_texts = []
            for path in path_pair:
                path_text = [data_manager.entity2text[path[0]], path[1], data_manager.entity2text[path[2]]]
                path_texts.append(' -> '.join(path_text))
            close_paths_text.append(f"{data_manager.entity2text[pos_head]}, {relation}, {data_manager.entity2text[pos_tail]}: {'; '.join(path_texts)}")
        
        neg_samples = data_manager.neg_sampling(pos_triple, 5)
        high_quality_neg_samples = high_quality_neg_sampling(data_manager, neg_samples)
        
        pos_reasoning_paths = "\n".join(
            " -> ".join(data_manager.triple_to_sentence(triple) for triple in path)
            for path in pos_paths
        )
        pos_reasoning_input = REASONING_PROMPT.format(reasoning_paths=pos_reasoning_paths, test_triple=data_manager.triple_to_sentence(pos_triple))
        pos_reasoning_output = "Y"
        sft_instructions.append({"instruction": pos_reasoning_input, "input": "", "output": pos_reasoning_output})
        for neg_triple in high_quality_neg_samples:
            neg_reasoning_paths = "\n".join(
                " -> ".join(data_manager.triple_to_sentence(triple) for triple in path)
                for path in bfs_paths(data_manager.entity2relationtail_dict, neg_triple[0], neg_triple[2], max_path_hop, max_paths)
            )
            neg_reasoning_input = REASONING_PROMPT.format(reasoning_paths=neg_reasoning_paths, test_triple=data_manager.triple_to_sentence(neg_triple))
            neg_reasoning_output = "N"
            sft_instructions.append({"instruction": neg_reasoning_input, "input": "", "output": neg_reasoning_output})
                    
    if setting == "inductive":
        close_path_dict_path = f"instructions/{dataset}/close_path.json"
        close_path_text_path = f"instructions/{dataset}/close_path_text.txt"
        sft_instructions_path = f"instructions/{dataset}/sft_instructions.json"
    else:
        close_path_dict_path = f"instructions/{dataset}/close_path_train_size_{train_size}.json"
        close_path_text_path = f"instructions/{dataset}/close_path_text_train_size_{train_size}.txt"
        sft_instructions_path = f"instructions/{dataset}/sft_instructions_train_size_{train_size}.json"
    
    with open(close_path_dict_path, "w", encoding="utf-8") as f:
        json.dump(close_path_dict, f, ensure_ascii=False, indent=4)
    
    with open(close_path_text_path, "w", encoding="utf-8") as file:
        for line in close_paths_text:
            file.write(line + "\n")
            
    with open(sft_instructions_path, "w", encoding="utf-8") as f:
        json.dump(sft_instructions, f, ensure_ascii=False, indent=4)
            
def main():
    parser = argparse.ArgumentParser(description='Process datasets with given hyperparameters')
    parser.add_argument("--dataset", type=str, choices=["FB15k-237-subset", "NELL-995-subset", "WN18RR-subset"], default="FB15k-237-subset")
    parser.add_argument('--max_path_hop', type=int, default=3)
    parser.add_argument('--max_paths', type=int, default=6)
    parser.add_argument("--train_size", type=str, choices=["full", "1000", "2000"], default="full", help="Size of the training data")

    args = parser.parse_args()
    build_instructions(args.dataset, args.max_path_hop, args.max_paths, args.train_size)

if __name__ == "__main__":
    main()