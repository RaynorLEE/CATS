import os
import json

from data_manager import DataManager
from tqdm import tqdm
import argparse

def build_instructions(dataset, train_size, neg_num):
    setting = "transductive" # 指令构建默认是transductive，用训练集
    
    data_manager = DataManager(dataset=dataset, setting=setting, train_size=train_size)

    paths_dir = f"instructions_v2/{dataset}"
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

        neg_samples = data_manager.neg_sampling(pos_triple, neg_num)
        for neg_triple in neg_samples:
            neg_ontology_prompt = data_manager.build_ontology_prompt(neg_triple)
            sft_instructions.append({"instruction": neg_ontology_prompt, "input": "", "output": "N"})
            
            neg_path_prompt = data_manager.build_path_prompt(neg_triple)
            sft_instructions.append({"instruction": neg_path_prompt, "input": "", "output": "N"})

        data_manager.entity2relationtail_dict[pos_head].append(removed_from_head)
        data_manager.entity2relationtail_dict[pos_tail].append(removed_from_tail)

    sft_instructions_path = f"{paths_dir}/{dataset}_train_size_{train_size}.json"
    with open(sft_instructions_path, "w", encoding="utf-8") as f:
        json.dump(sft_instructions, f, ensure_ascii=False, indent=4)
            
def main():
    parser = argparse.ArgumentParser(description='Process datasets with given hyperparameters')
    parser.add_argument("--dataset", type=str, choices=["FB15k-237-subset", "NELL-995-subset", "WN18RR-subset"], default="FB15k-237-subset")
    parser.add_argument("--train_size", type=str, choices=["full", "1000", "2000"], default="full", help="Size of the training data")
    parser.add_argument("--neg_num", type=int, default=3, help="Number of negative samples")

    args = parser.parse_args()
    build_instructions(args.dataset, args.train_size, args.neg_num)

if __name__ == "__main__":
    main()