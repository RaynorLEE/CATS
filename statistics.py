import json
import matplotlib.pyplot as plt

# 文件路径
json_file_path = '/home/yangcehao/hyper_inductive/datasets/FB15k-237-subset-inductive/paths/close_path.json'
txt_file_path = '/home/yangcehao/hyper_inductive/datasets/FB15k-237-subset-inductive/test.txt'

# 读取JSON文件
with open(json_file_path, 'r') as file:
    data = json.load(file)

# 初始化路径计数器
test_triple_path_count = {}
empty_path_count = 0

# 读取并处理test.txt文件
with open(txt_file_path, 'r') as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) >= 2:  # 确保至少有两个元素
            key = f"{parts[0]}-{parts[-1]}"
            # 检查这个key在JSON数据中是否存在
            if key in data:
                value = data[key]
                # 统计路径数量，如果是[]则计为0，否则为列表长度
                path_count = len(value) if value != [] else 0
                if path_count == 0:
                    empty_path_count += 1
                if path_count not in test_triple_path_count:
                    test_triple_path_count[path_count] = 0
                test_triple_path_count[path_count] += 1

# 准备分区间
bins = [0, 1, 2, 3, 4, 5, 10, 20, 50]
bin_labels = ['0', '1', '2', '3', '4', '5', '6-10', '11-20', '21-50', '51+']
bin_counts = {label: 0 for label in bin_labels}

# 根据bins对path_counts进行分区
for count, num in test_triple_path_count.items():
    if count > 50:
        bin_counts['51+'] += num
    else:
        for i in range(len(bins)):
            if count <= bins[i]:
                bin_counts[bin_labels[i]] += num
                break

# 绘制折线图，设置更低的高度
plt.figure(figsize=(5, 2))  # 修改此处的高度为4英寸
x_labels = list(bin_counts.keys())
y_values = list(bin_counts.values())

plt.plot(x_labels, y_values, marker='o', linestyle='-', color='skyblue')

# 添加y值标注
for i, y in enumerate(y_values):
    plt.text(i, y, str(y), ha='center', va='bottom')

plt.xticks(rotation=45)
plt.tight_layout()

plt.xlabel('#Path')
plt.gca().xaxis.set_label_coords(1.05, -0.05)  # Adjust the position as needed

# Move the ylabel to the top side of the y-axis
plt.ylabel('#Triple')
plt.gca().yaxis.set_label_coords(-0.05, 1.05)  # Adjust the position as needed, and rotate it
plt.gca().yaxis.label.set_rotation(0)
# plt.subplots_adjust(bottom=0.2, right=1.5)

# 保存图像到文件
from matplotlib.transforms import Bbox
plt.savefig('/home/yangcehao/path_count_bins_line_plot.pdf', 
            bbox_inches=Bbox.from_extents(-0.1, 0, 5.4, 2))
plt.show()