"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import wandb
import torch
import copy
import torch.nn as nn
from torch.utils import data
from pprint import pprint


from tqdm import tqdm
from ..datasets.urbancars import UrbanCars
from ..datasets.waterbirds import Waterbirds
from ..datasets.BFFHQ import BFFHQ
from ..datasets.dogs_and_cats import DogsAndCats
from ..datasets.CMNIST import CMNIST
from ..datasets.bar import BAR

from ..models.classifiers import (
    get_classifier,
    get_transforms,
)

from ..utils.reprod import set_seed
from ..utils.advanced_metrics import SubgroupMetricsTracker, eval_inacc_and_gap


class BaseTrainer:
    def __init__(self, args):
        self.args = args
        self._setup_method_name_and_default_name()

        self.cur_epoch = 1

        if args.run_name is None:
            args.run_name = self.default_name + f"_opt-{args.optimizer}_pt-{args.pretrain}_lr-{args.lr}_wd-{args.weight_decay}"
        else:
            args.run_name += f"_{self.default_name}"
        ckpt_dir = os.path.join(
            args.exp_root, args.run_name, f"seed_{args.seed}"
        )

        print("ckpt_dir: ", ckpt_dir)
        self.ckpt_dir = ckpt_dir

        self.ckpt_fname = "ckpt"

        self.cond_best_acc = 0
        self.cond_on_best_val_log_dict = {}
        pprint(vars(args))
        

    def _setup_all(self):
        args = self.args
        set_seed(args.seed)
        self.device = torch.device(0)

        self.scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

        self._setup_dataset()
        self._setup_models()
        self._setup_criterion()
        self._setup_optimizers()
        self._method_specific_setups()

        if args.wandb:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                name=args.run_name,
                config=args,
                settings=wandb.Settings(start_method="fork"),
            )
       
        self._before_train()

    def _get_train_collate_fn(self):
        return None

    def _get_train_loader(self, train_set):
        args = self.args
        if args.reweight_groups or args.reweight_classes:
            sampler = data.WeightedRandomSampler(self.train_set.get_sampling_weights(args.reweight_classes), num_samples=len(self.train_set), replacement=True)
            train_loader = data.DataLoader(
                self.train_set,
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=args.num_workers
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                self.train_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                persistent_workers=args.num_workers > 0,
                collate_fn=self._get_train_collate_fn(),
            )
        return train_loader

    def _setup_early_stop_metric(self):
        return 
        args = self.args

        early_stop_metric_arg_to_real_metric = {
            "bg": "val_bg_worst_group_acc",
            "co_occur_obj": "val_co_occur_obj_worst_group_acc",
            "both": "val_both_worst_group_acc",
        }

        if args.method in [
            "groupdro",
            "di",
            "subg",
            "dfr",
        ]:
            args.early_stop_metric_real = early_stop_metric_arg_to_real_metric[
                args.group_label
            ]
        elif args.method in [
            "erm",
            "lff",
            "eiil",
            "sd",
            "jtt",
            "debian",
            "lle",
            "cf_f_aug",
            "augmix",
            "cutmix",
            "mixup",
            "cutout",
        ]:
            # methods that do not use group labels
            args.early_stop_metric_real = early_stop_metric_arg_to_real_metric[
                args.early_stop_metric
            ]
        else:
            raise ValueError(f"unknown method: {args.method}")

    def _get_train_transform(self):
        args = self.args
        train_transform = get_transforms(args.arch, is_training=True)
        return train_transform

    def _setup_dataset(self):
        # TODO: Handle missing validation set
        args = self.args

        train_transform = self._get_train_transform()
        test_transform = get_transforms(args.arch, is_training=False)
        
        match args.dataset:
            
            case "waterbirds":
                self.train_set = Waterbirds(env="train", transform=train_transform, return_index=True)
                self.val_set   = Waterbirds(env="val", transform=test_transform, return_index=True)
                self.test_set  = Waterbirds(env="test", transform=test_transform, return_index=True)
                
            case "cmnist":
                self.train_set = CMNIST(env="train", bias_amount=args.rho)
                self.val_set   = CMNIST(env="val", bias_amount=args.rho)
                self.test_set  = CMNIST(env="test", bias_amount=args.rho)                
            
            case "bffhq":
                self.train_set = BFFHQ(env="train", bias_amount=args.rho, return_index=True, transform=train_transform)
                self.val_set   = BFFHQ(env="val", bias_amount=args.rho, return_index=True, transform=test_transform)
                self.test_set  = BFFHQ(env="test", bias_amount=args.rho, return_index=True, transform=test_transform)
               
            case "dogs_and_cats":
                self.train_set = DogsAndCats(env="train", bias_amount=args.rho, return_index=True, transform=train_transform)
                self.val_set   = DogsAndCats(env="val", bias_amount=args.rho, return_index=True, transform=test_transform)
                self.test_set  = DogsAndCats(env="test", bias_amount=args.rho, return_index=True, transform=test_transform)
            
            case "urbancars":
                self.train_set = UrbanCars(env="train", group_label="both", transform=train_transform)
                self.val_set   = UrbanCars(env="val", group_label="both", transform=test_transform)
                self.test_set  = UrbanCars(env="test", group_label="both", transform=test_transform)                
            
            case "bar":
                self.train_set = BAR(env="train", bias_amount=args.rho, return_index=True, transform=train_transform)
                self.val_set   = BAR(env="val", bias_amount=args.rho, return_index=True, transform=test_transform)
                self.test_set  = BAR(env="test", bias_amount=args.rho, return_index=True, transform=test_transform)
            
            case _:
                raise NotImplementedError()
            
        self.num_classes = self.train_set.num_classes
        self.num_groups = self.train_set.num_groups
        
        self.train_set = self._modify_train_set(self.train_set)
        train_loader = self._get_train_loader(self.train_set)
        val_loader   = data.DataLoader(
            self.val_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        test_loader = data.DataLoader(
            self.test_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers                    
        )
        
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.test_loader    = test_loader

    def _method_specific_setups(self):
        pass

    def _setup_models(self):
        translation_dict = {
            "none": None,
            "v1": "IMAGENET1K_V1",
            "default": "DEFAULT"
        }
        args = self.args
        
        self.classifier: nn.Module = get_classifier(
            args.arch,
            self.num_classes,
            weights = translation_dict[args.pretrain]
        ).to(self.device)

    def _set_train(self):
        self.classifier.train()

    def _setup_criterion(self):
        match self.args.dataset:
            case _:
                self.criterion = nn.CrossEntropyLoss()

    def _setup_optimizers(self):
        args = self.args
        parameters = [
            p for p in self.classifier.parameters() if p.requires_grad
        ]
        match args.optimizer:
            case "sgd":
                self.optimizer = torch.optim.SGD(
                    parameters,
                    args.lr,
                    nesterov=args.nesterov,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                )
            case "adam":
                self.optimizer = torch.optim.Adam(
                    parameters,
                    args.lr,
                    weight_decay=args.weight_decay
                )
            case "adamw":
                self.optimizer = torch.optim.AdamW(
                    parameters,
                    args.lr,
                    weight_decay=args.weight_decay
                )
            case _: raise NotImplementedError
                
                

    def _setup_method_name_and_default_name(self):
        raise NotImplementedError

    def _modify_train_set(self, train_dataset):
        return train_dataset

    def _before_train(self):
        pass

    def __call__(self):
        torch.backends.cudnn.benchmark = True
        args = self.args
        self._setup_all()

        for _ in range(self.cur_epoch, args.epoch + 1):
            self.train()
            # self.eval()
            self.cur_epoch += 1
            
        avg_acc, meter = self.eval()
        if args.dataset == "urbancars":
            inacc_gap_metric = eval_inacc_and_gap(self.train_set, meter)
            print(inacc_gap_metric)
            # wandb.log({
            #     "avg_acc": avg_acc,
            #     "inacc_gap_metric": inacc_gap_metric
            # }, step=self.cur_epoch, commit=False)
        
        os.makedirs(os.path.join("saved_models", self.args.dataset), exist_ok=True)
        torch.save(self.classifier.state_dict(), os.path.join("saved_models", self.args.dataset, "final_model.pt"))

    def train(self):
        raise NotImplementedError

    def eval(self):
        from ..erm_training import evaluate_model
        return evaluate_model(
            self.classifier, 
            test_loader=self.test_loader, 
            num_classes=self.num_classes, 
            num_groups=self.num_groups,
            criterion=torch.nn.CrossEntropyLoss(),
            epoch=self.cur_epoch,
            device=self.device,
            wb=wandb if self.args.wandb else None,
            prefix="te",
            config=self.args
        )

    @torch.no_grad()
    def _eval_split(self, loader, split):
        raise NotImplementedError()

    def _state_dict_for_save(self):
        state_dict = {
            "classifier": self.classifier.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.cur_epoch,
        }
        return state_dict

    def _load_state_dict(self, state_dict):
        self.scaler.load_state_dict(state_dict["scaler"])
        self.cur_epoch = state_dict["epoch"] + 1
        self.classifier.load_state_dict(state_dict["classifier"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.cond_best_acc = state_dict["cond_best_acc"]
        self.cond_on_best_val_log_dict = state_dict["cond_on_best_val_log_dict"]

    def _save_ckpt(self, state_dict, name):
        ckpt_fpath = os.path.join(self.ckpt_dir, f"{name}.pth")
        torch.save(state_dict, ckpt_fpath)

    def _loss_backward(self, loss, retain_graph=False):
        if self.args.amp:
            self.scaler.scale(loss).backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)

    def _optimizer_step(self, optimizer):
        if self.args.amp:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def _scaler_update(self):
        if self.args.amp:
            self.scaler.update()

    def checkpoint(self):
        new_args = copy.deepcopy(self.args)
        ckpt_fpath = os.path.join(self.ckpt_dir, f"{self.ckpt_fname}.pth")
        # if os.path.exists(ckpt_fpath):
        #     new_args.resume = ckpt_fpath

    def log_to_wandb(self, log_dict, step=None):
        if step is None:
            step = self.cur_epoch
        if self.args.wandb:
            wandb.log(log_dict, step=step)
