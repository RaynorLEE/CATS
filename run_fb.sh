CUDA_VISIBLE_DEVICES=4 python prediction.py --setting inductive --dataset FB15k-237-subset --train_size 1000
CUDA_VISIBLE_DEVICES=4 python prediction.py --setting inductive --dataset FB15k-237-subset --train_size 2000
CUDA_VISIBLE_DEVICES=4 python prediction.py --setting transductive --dataset FB15k-237-subset --train_size 1000
CUDA_VISIBLE_DEVICES=4 python prediction.py --setting transductive --dataset FB15k-237-subset --train_size 2000