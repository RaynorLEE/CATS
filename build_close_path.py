import os
import json
from collections import defaultdict, deque
from data_manager import DataManager
from tqdm import tqdm
import argparse

def bfs_paths(entity2relationtail_dict, start, goal, max_hops):
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
    return paths

def process_dataset(dataset, setting, max_path_hop, train_size):
    data_manager = DataManager(dataset=dataset, setting=setting, train_size=train_size)
    entity2relationtail_dict = data_manager.entity2relationtail_dict

    paths_dir = f"{data_manager.dataset_path}/paths"
    os.makedirs(paths_dir, exist_ok=True)

    close_path_dict = {}
    close_paths_text = []
    
    for triple in tqdm(data_manager.test_set, desc=f"Processing {dataset} - setting: {setting} - Train_size: {train_size}"):
        head, relation, tail = triple
        paths = list(bfs_paths(entity2relationtail_dict, head, tail, max_path_hop))
        close_path_dict[f"{head}-{tail}"] = paths
        
        for path_pair in paths:
            path_texts = []
            for path in path_pair:
                path_text = [data_manager.entity2text[path[0]], path[1], data_manager.entity2text[path[2]]]
                path_texts.append(' -> '.join(path_text))
            close_paths_text.append(f"{data_manager.entity2text[head]}, {data_manager.entity2text[tail]}: {'; '.join(path_texts)}")

    if setting == "inductive":
        close_path_dict_path = f"{data_manager.dataset_path}/paths/close_path.json"
        close_path_text_path = f"{data_manager.dataset_path}/paths/close_path_text.txt"
    else:
        close_path_dict_path = f"{data_manager.dataset_path}/paths/close_path_train_size_{train_size}.json"
        close_path_text_path = f"{data_manager.dataset_path}/paths/close_path_text_train_size_{train_size}.txt"
    
    with open(close_path_dict_path, "w", encoding="utf-8") as f:
        json.dump(close_path_dict, f, ensure_ascii=False, indent=4)
    
    with open(close_path_text_path, "w", encoding="utf-8") as file:
        for line in close_paths_text:
            file.write(line + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["FB15k-237-subset", "NELL-995-subset", "WN18RR-subset"], default="FB15k-237-subset")
    parser.add_argument("--setting", type=str, choices=["inductive", "transductive"], default="inductive", help="Inductive or Transductive setting")
    parser.add_argument('--max_path_hop', type=int, default=3)
    parser.add_argument("--train_size", type=str, choices=["full", "1000", "2000"], default="full", help="Size of the training data")

    args = parser.parse_args()
    process_dataset(args.dataset, args.setting, args.max_path_hop, args.train_size)

if __name__ == "__main__":
    main()