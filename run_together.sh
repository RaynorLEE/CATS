CUDA_VISIBLE_DEVICES=1 python prediction.py --setting inductive --dataset FB15k-237-subset
CUDA_VISIBLE_DEVICES=1 python prediction.py --setting transductive --dataset FB15k-237-subset
# CUDA_VISIBLE_DEVICES=1 python prediction.py --setting inductive --dataset WN18RR-subset
# CUDA_VISIBLE_DEVICES=1 python prediction.py --setting transductive --dataset WN18RR-subset
# CUDA_VISIBLE_DEVICES=1 python prediction.py --setting inductive --dataset NELL-995-subset
# CUDA_VISIBLE_DEVICES=1 python prediction.py --setting transductive --dataset NELL-995-subset