import os
import json
import random
from collections import defaultdict, deque
from data_manager import DataManager
from tqdm import tqdm
import argparse

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

def close_path_finder(data_manager: DataManager, head, tail):
    close_paths = list(bfs_paths(data_manager.entity2relationtail_dict, head, tail, data_manager.max_path_hops, data_manager.max_paths))
    if close_paths:
        path_degrees = []
        for path in close_paths:
            degree_sum = sum(data_manager.relation_degree_dict[relation] for _, relation, _ in path)
            path_degrees.append((degree_sum, path))
        path_degrees.sort(key=lambda x: x[0])

        top_paths = [path for _, path in path_degrees[:data_manager.max_paths]]
        top_paths.reverse()
        return top_paths
    return []

def build_instructions(dataset, train_size):
    setting = "transductive" # 指令构建默认是transductive，用训练集
    
    data_manager = DataManager(dataset=dataset, setting=setting, train_size=train_size)

    paths_dir = f"instructions/{dataset}"
    os.makedirs(paths_dir, exist_ok=True)

    sft_instructions = []
    
    for pos_triple in tqdm(data_manager.path_set, desc=f"Processing {dataset} - setting: {setting} - Train_size: {train_size}"):
        pos_head, relation, pos_tail = pos_triple
        
        # 移除当前triple，因为是寻找当前(pos_head, pos_tail)的close_path
        removed_from_head = (relation, pos_tail, 1)
        removed_from_tail = (relation, pos_head, -1)
        data_manager.entity2relationtail_dict[pos_head].remove(removed_from_head)
        data_manager.entity2relationtail_dict[pos_tail].remove(removed_from_tail)
        
        pos_ontology_prompt = data_manager.build_ontology_prompt(pos_triple)
        sft_instructions.append({"instruction": pos_ontology_prompt, "input": "", "output": "Y"})
        
        pos_path_prompt = data_manager.build_path_prompt(pos_triple)
        sft_instructions.append({"instruction": pos_path_prompt, "input": "", "output": "Y"})

        neg_samples = data_manager.neg_sampling(pos_triple, 3)
        for neg_triple in neg_samples:
            neg_ontology_prompt = data_manager.build_ontology_prompt(neg_triple)
            sft_instructions.append({"instruction": neg_ontology_prompt, "input": "", "output": "N"})
            
            neg_path_prompt = data_manager.build_path_prompt(neg_triple)
            sft_instructions.append({"instruction": neg_path_prompt, "input": "", "output": "N"})

        data_manager.entity2relationtail_dict[pos_head].append(removed_from_head)
        data_manager.entity2relationtail_dict[pos_tail].append(removed_from_tail)
                    
    sft_instructions_path = f"instructions/{dataset}/{dataset}_train_size_{train_size}.json"
    with open(sft_instructions_path, "w", encoding="utf-8") as f:
        json.dump(sft_instructions, f, ensure_ascii=False, indent=4)
            
def main():
    parser = argparse.ArgumentParser(description='Process datasets with given hyperparameters')
    parser.add_argument("--dataset", type=str, choices=["FB15k-237-subset", "NELL-995-subset", "WN18RR-subset"], default="FB15k-237-subset")
    parser.add_argument("--train_size", type=str, choices=["full", "1000", "2000"], default="full", help="Size of the training data")

    args = parser.parse_args()
    build_instructions(args.dataset, args.train_size)

if __name__ == "__main__":
    main()