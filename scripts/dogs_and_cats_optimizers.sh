#!/bin/bash

for pretrain in default none; do
    for optimizer in sgd adam adamw; do
        for rho in 95 99; do
            for start_seed in 0 1 2; do
                python main.py \
                    --method erm \
                    --arch resnet18 \
                    --wandb \
                    --wandb_project_name MindYourDefaults \
                    --amp \
                    --pretrain $pretrain \
                    --lr 1e-04 \
                    --optimizer $optimizer \
                    --dataset dogs_and_cats \
                    --epoch 60 \
                    --batch_size=128 \
                    --rho $rho \
                    --weight_decay 1e-04 \
                    --reweight_classes \
                    --start_seed $start_seed
            done
        done
    done
done