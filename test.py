# 打开/data/FinAi_Mapping_Knowledge/yangcehao/hyper_inductive/data/FB15k-237-subset-inductive/test.txt，遍历

# 打开close_path_dict.json，读入为close_path_dict
import json
with open('data/FB15k-237-subset-inductive/close_path_dict.json', 'r', encoding='utf-8') as f:
    close_path_dict = json.load(f)

with open('data/FB15k-237-subset-inductive/test.txt', 'r', encoding='utf-8') as f:
    for line in f:
        head, relation, tail = line.strip().split('\t')
        key = f'{head}-{tail}'
        print(head, relation, tail)
        # print(close_path_dict[key])
        if close_path_dict[key] == []:
            print('empty')