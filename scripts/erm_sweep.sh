#!/bin/bash



python main.py \
    --method erm \
    --wandb \
    --amp \
    --pretrain default \
    --lr 1e-04 \
    --optimizer sgd \
    --dataset vas \
    --rho 95

python main.py \
    --method erm \
    --wandb \
    --amp \
    --pretrain default \
    --lr 1e-04 \
    --optimizer sgd \
    --dataset vas \
    --rho 98

python main.py \
    --method erm \
    --wandb \
    --amp \
    --pretrain default \
    --lr 1e-04 \
    --optimizer sgd \
    --dataset vas \
    --rho 99.5


python main.py \
    --method erm \
    --wandb \
    --amp \
    --pretrain default \
    --lr 1e-04 \
    --optimizer sgd \
    --dataset vas \
    --rho 100