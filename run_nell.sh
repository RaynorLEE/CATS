# CUDA_VISIBLE_DEVICES=4 python prediction.py --setting inductive --dataset NELL-995-subset
# CUDA_VISIBLE_DEVICES=4 python prediction.py --setting transductive --dataset NELL-995-subset
CUDA_VISIBLE_DEVICES=4 python prediction.py --setting inductive --dataset NELL-995-subset --train_size 1000
CUDA_VISIBLE_DEVICES=4 python prediction.py --setting transductive --dataset NELL-995-subset --train_size 1000