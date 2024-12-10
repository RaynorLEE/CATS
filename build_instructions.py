import os
import json

from data_manager import DataManager
from tqdm import tqdm
import argparse
from prompt_templates import SUBGRAPH_REASON_PROMPT, NEIGHBOR_REASON_PROMPT, CLOSE_PATH_REASON_PROMPT

# 用一个relation_degree计算所有close_paths的degree和，然后排序，取最小的几个，这样能排除"gender","ethnicity"等高频relation
def close_path_finder(data_manager:DataManager, triple):
    head, relation, tail = triple
    close_paths = list(data_manager.bfs_paths(head, tail))

    if close_paths:
        path_degrees = []
        for path in close_paths:
            degree_sum = sum(data_manager.relation_degree_dict[rel] for _, rel, _ in path)
            path_degrees.append((degree_sum, path))
        path_degrees.sort(key=lambda x: x[0])
        
        top_paths = [path for _, path in path_degrees[:data_manager.max_reason_paths]]
        top_paths.reverse()
        return top_paths

    return []

def build_instructions(dataset, train_size, subgraph_type, neg_num, version):
    setting = "transductive" # 指令构建默认是transductive，用训练集
    
    data_manager = DataManager(dataset=dataset, setting=setting, train_size=train_size)

    paths_dir = f"instructions{version}/{dataset}"
    os.makedirs(paths_dir, exist_ok=True)

    sft_instructions = []
    
    for pos_triple in tqdm(data_manager.path_set, desc=f"Processing {dataset} - setting: {setting} - Train_size: {train_size}"):
        pos_head, relation, pos_tail = pos_triple
        
        # 移除当前triple，因为是寻找当前(pos_head, pos_tail)的close_path
        removed_from_head = (relation, pos_tail, 1)
        removed_from_tail = (relation, pos_head, -1)
        data_manager.entity2relationtail_dict[pos_head].remove(removed_from_head)
        data_manager.entity2relationtail_dict[pos_tail].remove(removed_from_tail)
        
        pos_type_prompt = data_manager.build_type_prompt(pos_triple)
        sft_instructions.append({"instruction": pos_type_prompt, "input": "", "output": "Y"})
        
        if subgraph_type == "combine":
            pos_neighbor_triples = data_manager.neighbor_triple_finder(pos_triple)
            pos_close_paths = close_path_finder(data_manager, pos_triple)
            pos_reasoning_paths = "\n".join(
                " -> ".join(data_manager.triple_to_sentence(triple) for triple in path)
                for path in pos_close_paths
            )
            pos_subgraph_prompt = SUBGRAPH_REASON_PROMPT.format(neighbor_triples="\n".join(pos_neighbor_triples), reasoning_paths=pos_reasoning_paths, test_triple=data_manager.triple_to_sentence(pos_triple))
        elif subgraph_type == "neighbor-only":
            pos_neighbor_triples = data_manager.neighbor_triple_finder(pos_triple)
            pos_subgraph_prompt = NEIGHBOR_REASON_PROMPT.format(neighbor_triples="\n".join(pos_neighbor_triples), test_triple=data_manager.triple_to_sentence(pos_triple))
        elif subgraph_type == "path-only":
            pos_close_paths = close_path_finder(data_manager, pos_triple)
            pos_reasoning_paths = "\n".join(
                " -> ".join(data_manager.triple_to_sentence(triple) for triple in path)
                for path in pos_close_paths
            )
            pos_subgraph_prompt = CLOSE_PATH_REASON_PROMPT.format(reasoning_paths=pos_reasoning_paths, test_triple=data_manager.triple_to_sentence(pos_triple))
            
        sft_instructions.append({"instruction": pos_subgraph_prompt, "input": "", "output": "Y"})

        neg_samples = data_manager.neg_sampling(pos_triple, neg_num)
        for neg_triple in neg_samples:
            neg_type_prompt = data_manager.build_type_prompt(neg_triple)
            sft_instructions.append({"instruction": neg_type_prompt, "input": "", "output": "N"})
            
            if subgraph_type == "combine":
                neg_neighbor_triples = data_manager.neighbor_triple_finder(neg_triple)
                neg_close_paths = close_path_finder(data_manager, neg_triple)
                neg_reasoning_paths = "\n".join(
                    " -> ".join(data_manager.triple_to_sentence(triple) for triple in path)
                    for path in neg_close_paths
                )
                neg_subgraph_prompt = SUBGRAPH_REASON_PROMPT.format(neighbor_triples="\n".join(neg_neighbor_triples), reasoning_paths=neg_reasoning_paths, test_triple=data_manager.triple_to_sentence(neg_triple))
            elif subgraph_type == "neighbor-only":
                neg_neighbor_triples = data_manager.neighbor_triple_finder(neg_triple)
                neg_subgraph_prompt = NEIGHBOR_REASON_PROMPT.format(neighbor_triples="\n".join(neg_neighbor_triples), test_triple=data_manager.triple_to_sentence(neg_triple))
            elif subgraph_type == "path-only":
                neg_close_paths = close_path_finder(data_manager, neg_triple)
                neg_reasoning_paths = "\n".join(
                    " -> ".join(data_manager.triple_to_sentence(triple) for triple in path)
                    for path in neg_close_paths
                )
                neg_subgraph_prompt = CLOSE_PATH_REASON_PROMPT.format(reasoning_paths=neg_reasoning_paths, test_triple=data_manager.triple_to_sentence(neg_triple))
                
            sft_instructions.append({"instruction": neg_subgraph_prompt, "input": "", "output": "N"})

        data_manager.entity2relationtail_dict[pos_head].append(removed_from_head)
        data_manager.entity2relationtail_dict[pos_tail].append(removed_from_tail)

    sft_instructions_path = f"{paths_dir}/{dataset}_train_size_{train_size}_{subgraph_type}.json"
    with open(sft_instructions_path, "w", encoding="utf-8") as f:
        json.dump(sft_instructions, f, ensure_ascii=False, indent=4)
        
def build_vanilla_instructions(dataset, train_size, neg_num, version):
    setting = "transductive" # 指令构建默认是transductive，用训练集
    
    data_manager = DataManager(dataset=dataset, setting=setting, train_size=train_size)

    paths_dir = f"vanilla_instructions{version}/{dataset}"
    os.makedirs(paths_dir, exist_ok=True)

    sft_instructions = []
    
    for pos_triple in tqdm(data_manager.path_set, desc=f"Processing {dataset} - setting: {setting} - Train_size: {train_size}"):
        pos_vanilla_prompt = data_manager.build_vanilla_prompt(pos_triple)
        sft_instructions.append({"instruction": pos_vanilla_prompt, "input": "", "output": "Y"})
        neg_samples = data_manager.neg_sampling(pos_triple, neg_num)
        
        for neg_triple in neg_samples:
            neg_vanilla_prompt = data_manager.build_vanilla_prompt(neg_triple)
            sft_instructions.append({"instruction": neg_vanilla_prompt, "input": "", "output": "N"})
            
    sft_instructions_path = f"{paths_dir}/{dataset}_train_size_{train_size}.json"
    with open(sft_instructions_path, "w", encoding="utf-8") as f:
        json.dump(sft_instructions, f, ensure_ascii=False, indent=4)
            
def main():
    parser = argparse.ArgumentParser(description='Process datasets with given hyperparameters')
    parser.add_argument("--dataset", type=str, choices=["FB15k-237-subset", "NELL-995-subset", "WN18RR-subset"], default="FB15k-237-subset")
    parser.add_argument("--train_size", type=str, choices=["full", "1000", "2000"], default="full", help="Size of the training data")
    parser.add_argument("--neg_num", type=int, default=3, help="Number of negative samples")
    parser.add_argument("--prompt_type", type=str, default="CATS", choices=["CATS", "vanilla"])
    parser.add_argument("--subgraph_type", type=str, default="combine", choices=["neighbor-only", "path-only", "combine"])
    parser.add_argument("--version", type=str, default="")

    args = parser.parse_args()
    if args.prompt_type == "CATS":
        build_instructions(args.dataset, args.train_size, args.subgraph_type, args.neg_num, args.version)
    elif args.prompt_type == "vanilla":
        build_vanilla_instructions(args.dataset, args.train_size, args.neg_num, args.version)

if __name__ == "__main__":
    main()