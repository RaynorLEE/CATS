import os
import json
from collections import defaultdict, deque
from data_manager import DataManager
from tqdm import tqdm
import argparse

def process_dataset(dataset, setting, train_size, max_path_hops):
    data_manager = DataManager(dataset=dataset, setting=setting, train_size=train_size)
    data_manager.max_path_hops = max_path_hops

    paths_dir = f"{data_manager.dataset_path}/paths_{max_path_hops}hop"
    os.makedirs(paths_dir, exist_ok=True)

    close_path_dict = {}
    close_paths_text = []
    
    for triple in tqdm(data_manager.test_set, desc=f"Processing {dataset} - setting: {setting} - Train_size: {train_size}"):
        head, relation, tail = triple
        paths = list(data_manager.bfs_paths(head, tail))
        close_path_dict[f"{head}-{tail}"] = paths
        
        for path_pair in paths:
            path_texts = []
            for path in path_pair:
                path_text = [data_manager.entity2text[path[0]], path[1], data_manager.entity2text[path[2]]]
                path_texts.append(' -> '.join(path_text))
            close_paths_text.append(f"{data_manager.entity2text[head]}, {data_manager.entity2text[tail]}: {'; '.join(path_texts)}")

    if setting == "inductive":
        close_path_dict_path = f"{paths_dir}/close_path.json"
        close_path_text_path = f"{paths_dir}/close_path_text.txt"
    else:
        close_path_dict_path = f"{paths_dir}/close_path_train_size_{train_size}.json"
        close_path_text_path = f"{paths_dir}/close_path_text_train_size_{train_size}.txt"
    
    with open(close_path_dict_path, "w", encoding="utf-8") as f:
        json.dump(close_path_dict, f, ensure_ascii=False, indent=4)
    
    with open(close_path_text_path, "w", encoding="utf-8") as file:
        for line in close_paths_text:
            file.write(line + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["FB15k-237-subset", "NELL-995-subset", "WN18RR-subset"], default="FB15k-237-subset")
    parser.add_argument("--setting", type=str, choices=["inductive", "transductive"], default="inductive", help="Inductive or Transductive setting")
    parser.add_argument("--train_size", type=str, choices=["full", "1000", "2000"], default="full", help="Size of the training data")
    parser.add_argument("--max_path_hops", type=int, default=3, help="Maximum number of hops in the path")

    args = parser.parse_args()
    process_dataset(args.dataset, args.setting, args.train_size, args.max_path_hops)

if __name__ == "__main__":
    main()