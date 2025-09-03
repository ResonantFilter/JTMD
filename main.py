#!/usr/bin/env python3

"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
"""
Customization from XXXXXXXXXXXXXXXXXX
2025
"""


import argparse
import copy
import os


from methods import method_to_trainer
from utils.reprod import set_seed
from utils.wandb_wrapper import slurm_wandb_argparser
import wandb


def parse_args():
    parser = argparse.ArgumentParser(parents=[slurm_wandb_argparser()])
    parser.add_argument(
        "--method",
        type=str,
        choices=[
            "erm",
            "lff",
            "eiil",
            "sd",
            "groupdro",
            "di",
            "debian",
            "jtt",
            "subg",
            "lle",
            "cf_f_aug",
            "augmix",
            "mixup",
            "cutmix",
            "cutout",
            "dfr",
        ],
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="vas",
    )
    parser.add_argument("--rho", type=float, default=95, required=True)
    parser.add_argument("--bg_ratio", type=float, default=0.95)
    parser.add_argument("--co_occur_obj_ratio", type=float, default=0.95)

    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--num_seed", type=int, default=1)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--pretrain", type=str, default="default", choices=["none", "v1", "default"])
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--reweight_groups", action="store_true")
    parser.add_argument("--reweight_classes", action="store_true")

    parser.add_argument(
        "--exp_root", type=str, default="exp"
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", type=str)
    parser.add_argument("--early_stop_metric", type=str, default="both")
    parser.add_argument("--early_stop_metric_list", type=str, nargs="+")

    # EIIL
    parser.add_argument("--eiil_n_steps", type=int, default=10000)

    # Gradient Starvation
    parser.add_argument(
        "--sp",
        type=float,
        default=0.1,
        help="coefficient of logits norm penalty in spectral decoupling",
    )

    # GroupDRO
    parser.add_argument("--groupdro_robust_step_size", type=float, default=0.01)
    parser.add_argument("--groupdro_gamma", type=float, default=0.1)

    # shared by GroupDRO and Domain Independent
    parser.add_argument(
        "--group_label",
        type=str,
        choices=["bg", "co_occur_obj", "both"],
        default="both",
    )
    parser.add_argument("--group_label_list", type=str, nargs="+")

    # JTT
    parser.add_argument("--jtt_up_weight", type=int, default=100)

    # shared by EIIL and JTT
    parser.add_argument("--bias_id_epoch", type=int, default=1)

    # Mixup, Cutmix, Cutout
    parser.add_argument("--mixup_alpha", type=float, default=0.1)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--cutout", type=float, default=0.1)

    args = parser.parse_args()

    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./saved_models", exist_ok=True)
    os.makedirs(".logs/args.exp_root", exist_ok=True)

    if args.wandb:
        assert args.wandb_project_name is not None
        assert args.wandb_entity is not None

    return args


def main():
    args = parse_args()

    Trainer = method_to_trainer[args.method]

    group_label_list = (
        args.group_label_list if args.group_label_list else [args.group_label]
    )

    if args.early_stop_metric_list is not None:
        early_stop_metric_list = args.early_stop_metric_list
    else:
        early_stop_metric_list = [args.early_stop_metric]

    args_list = []
    for group_label in group_label_list:
        for seed in range(
            args.start_seed, args.start_seed + args.num_seed
        ):
            for early_stop_metric in early_stop_metric_list:
                new_args = copy.deepcopy(args)
                new_args.group_label = group_label
                new_args.seed = seed
                if (
                    args.method
                    in [
                        "groupdro",
                        "di",
                        "subg",
                        "dfr",
                    ]
                    and early_stop_metric != group_label
                ):
                    continue
                new_args.early_stop_metric = early_stop_metric
                args_list.append(new_args)

   
    for job_args in args_list:
        trainer = Trainer(job_args)
        trainer()
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
