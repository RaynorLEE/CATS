CUDA_VISIBLE_DEVICES=5 python prediction.py --setting inductive --dataset WN18RR-subset --train_size 1000
CUDA_VISIBLE_DEVICES=5 python prediction.py --setting inductive --dataset WN18RR-subset --train_size 2000
CUDA_VISIBLE_DEVICES=5 python prediction.py --setting transductive --dataset WN18RR-subset --train_size 1000
CUDA_VISIBLE_DEVICES=5 python prediction.py --setting transductive --dataset WN18RR-subset --train_size 2000