import os
import json

from data_manager import DataManager
from tqdm import tqdm
import argparse
from prompt_templates import EXPLAINING_PROMPT

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

def build_instructions(dataset, train_size, neg_num):
    setting = "transductive" 
    
    data_manager = DataManager(dataset=dataset, setting=setting, train_size=train_size)

    paths_dir = f"instructions_explain/{dataset}"
    os.makedirs(paths_dir, exist_ok=True)

    sft_instructions = []

    
    for pos_triple in tqdm(data_manager.path_set[:2000], desc=f"Processing {dataset} - setting: {setting} - Train_size: {train_size}"):
        pos_head, relation, pos_tail = pos_triple
        
        removed_from_head = (relation, pos_tail, 1)
        removed_from_tail = (relation, pos_head, -1)
        data_manager.entity2relationtail_dict[pos_head].remove(removed_from_head)
        data_manager.entity2relationtail_dict[pos_tail].remove(removed_from_tail)
        
        pos_neighbor_triples = data_manager.neighbor_triple_finder(pos_triple)
        pos_close_paths = close_path_finder(data_manager, pos_triple)
        pos_reasoning_paths = "\n".join(
            " -> ".join(data_manager.triple_to_sentence(triple) for triple in path)
            for path in pos_close_paths
        )
        pos_explain_prompt = EXPLAINING_PROMPT.format(neighbor_triples="\n".join(pos_neighbor_triples), reasoning_paths=pos_reasoning_paths, test_triple=data_manager.triple_to_sentence(pos_triple))
        pos_explain_output = ""
        sft_instructions.append({"instruction": pos_explain_prompt, "input": "", "output": pos_explain_prompt})

        neg_samples = data_manager.neg_sampling(pos_triple, neg_num)
        for neg_triple in neg_samples:
            
            neg_neighbor_triples = data_manager.neighbor_triple_finder(neg_triple)
            neg_close_paths = close_path_finder(data_manager, neg_triple)
            neg_reasoning_paths = "\n".join(
                " -> ".join(data_manager.triple_to_sentence(triple) for triple in path)
                for path in neg_close_paths
            )
            neg_explain_prompt = EXPLAINING_PROMPT.format(neighbor_triples="\n".join(neg_neighbor_triples), reasoning_paths=neg_reasoning_paths, test_triple=data_manager.triple_to_sentence(neg_triple))
            neg_explain_output = ""
            sft_instructions.append({"instruction": neg_explain_prompt, "input": "", "output": neg_explain_output})

        data_manager.entity2relationtail_dict[pos_head].append(removed_from_head)
        data_manager.entity2relationtail_dict[pos_tail].append(removed_from_tail)

    sft_instructions_path = f"{paths_dir}/{dataset}_train_size_{train_size}.json"
    with open(sft_instructions_path, "w", encoding="utf-8") as f:
        json.dump(sft_instructions, f, ensure_ascii=False, indent=4)
            
def main():
    parser = argparse.ArgumentParser(description='Process datasets with given hyperparameters')
    parser.add_argument("--dataset", type=str, choices=["FB15k-237-subset", "NELL-995-subset", "WN18RR-subset"], default="FB15k-237-subset")
    parser.add_argument("--train_size", type=str, choices=["full"], default="full", help="Size of the training data")
    parser.add_argument("--neg_num", type=int, default=1, help="Number of negative samples")

    args = parser.parse_args()
    build_instructions(args.dataset, args.train_size, args.neg_num)

if __name__ == "__main__":
    main()
