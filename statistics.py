import json

# 文件路径
train_file = '/home/yangcehao/hyper_inductive/datasets/NELL-995-subset/train_1000.txt'
test_file = '/home/yangcehao/hyper_inductive/datasets/NELL-995-subset/test.txt'

# 初始化meta_relation_count字典
meta_relation_count = {}

# 读取train1000.txt文件并统计meta_relation
with open(train_file, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split('\t')
        
        if len(parts) != 3:
            continue  # 确保数据格式正确
        
        head, relation, tail = parts
        
        # 获取head_type和tail_type
        head_type = head.split(':')[1]
        tail_type = tail.split(':')[1]
        
        # 构建meta_relation
        meta_relation = f'{head_type}-{relation}-{tail_type}'
        
        # 统计meta_relation的数量
        if meta_relation in meta_relation_count:
            meta_relation_count[meta_relation] += 1
        else:
            meta_relation_count[meta_relation] = 1

# 初始化fail_count
fail_count = 0

# 读取test.txt文件并检查meta_relation
with open(test_file, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split('\t')
        
        if len(parts) != 3:
            continue  # 确保数据格式正确
        
        head, relation, tail = parts
        
        # 获取head_type和tail_type
        head_type = head.split(':')[1]
        tail_type = tail.split(':')[1]
        
        # 构建meta_relation
        meta_relation = f'{head_type}-{relation}-{tail_type}'
        
        # 检查meta_relation是否在meta_relation_count中
        if meta_relation not in meta_relation_count or meta_relation_count[meta_relation] < 1:
            fail_count += 1

# 输出fail_count
print(f"Fail count: {fail_count}")
