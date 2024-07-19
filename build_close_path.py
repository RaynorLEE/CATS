import json
from collections import defaultdict, deque
from data_manager import DataManager
from tqdm import tqdm

# datasets = ["FB15k-237-subset", "NELL-995-subset", "WN18RR-subset"]
# inductives = ["True", "False"]

datasets = ["FB15k-237-subset"]
inductives = ["True"]
directions = ["head"]
for dataset in datasets:
    for inductive in inductives:
        for direction in directions:
            data_manager = DataManager(dataset=dataset, inductive=inductive, direction=direction)

            inductive_set = data_manager.inductive_set

            entity2relationtail_dict = defaultdict(list)
            for head, relation, tail in inductive_set:
                entity2relationtail_dict[head].append((relation, tail, 1))
                entity2relationtail_dict[tail].append((relation, head, -1))
            with open(f"{data_manager.dataset_path}/entity2relationtail_dict.json", "w", encoding="utf-8") as f:
                json.dump(entity2relationtail_dict, f, ensure_ascii=False, indent=4)

            def bfs_paths(entity2relationtail_dict, start, goal, max_hops):
                queue = deque([(start, [], 0)])
                paths = []
                while queue:
                    current, path, hops = queue.popleft()
                    if hops < max_hops:
                        for relation, neighbor, direction in entity2relationtail_dict[current]:
                            if direction == -1:
                                head, tail = neighbor, current
                            else:
                                head, tail = current, neighbor
                            new_path = path + [(head, relation, tail)]
                            if neighbor == goal:
                                paths.append(new_path)
                            else:
                                queue.append((neighbor, new_path, hops + 1))
                return paths

            max_path_hop = 5
            close_path_dict = {}
            for triple in tqdm(data_manager.test_set, desc=f"Processing {dataset} - Inductive: {inductive}"):
                head, relation, tail = triple
                paths = list(bfs_paths(entity2relationtail_dict, head, tail, max_path_hop))
                close_path_dict[f"{head}-{tail}"] = paths

            with open(f"{data_manager.dataset_path}/close_path_ranking_{direction}.json", "w", encoding="utf-8") as f:
                json.dump(close_path_dict, f, ensure_ascii=False, indent=4)

            print("Process completed and files saved.")
