#!/bin/bash

for method in erm lff; do

    python main.py \
        --method $method \
        --arch mlp \
        --wandb \
        --wandb_project_name PD \
        --pretrain none \
        --lr 1e-04 \
        --optimizer adam \
        --amp \
        --dataset cmnist \
        --epoch 100 \
        --batch_size=256 \
        --rho 95 \
        --weight_decay 0 \
        --start_seed 0
done