"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from GroupDRO:
# https://github.com/kohpangwei/group_DRO
# --------------------------------------------------------

import torch
import torch.nn as nn


from tqdm import tqdm
import wandb
from .base_trainer import BaseTrainer
from ..utils.advanced_metrics import AverageMeter, SubgroupMetricsTracker
from torch.utils.data.sampler import WeightedRandomSampler


class GroupDROTrainer(BaseTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "groupdro"
        default_name = f"{args.method}_{args.group_label}_{args.dataset}"
        self.default_name = default_name

    def _get_train_loader(self, train_set):
        args = self.args
        weights = train_set.get_sampling_weights()
        sampler = WeightedRandomSampler(weights, len(train_set), replacement=True)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            sampler=sampler,
            persistent_workers=args.num_workers > 0,
        )
        return train_loader

    def _setup_criterion(self):
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def _method_specific_setups(self):
        num_group = self.train_set.num_groups
        self.num_groups = num_group
        self.adv_probs = torch.ones(num_group, device=self.device) / num_group
        self.group_range = torch.arange(
            num_group, dtype=torch.long, device=self.device
        ).unsqueeze(1)

    def train(self):
        args = self.args
        self.classifier.train()
        losses = AverageMeter()
        meter = SubgroupMetricsTracker(
            num_classes=self.num_classes,
            num_groups=8,
            device=self.device,
            log_history=True,
            wandb_logger = wandb if self.args.wandb else None,
            prefix="tr"
        )

        pbar = tqdm(self.train_loader, dynamic_ncols=True)
        for batch, (dat, labels, _) in enumerate(pbar):
            image, target = dat, labels
            obj_gt = target[0]  
            group_gt = labels[1]
            image = image.to(self.device, non_blocking=True)
            obj_gt = obj_gt.to(self.device, non_blocking=True)
            group_gt = group_gt.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=args.amp):
                output = self.classifier(image)
                loss_per_sample = self.criterion(output, obj_gt)

                # compute group loss
                group_map = (group_gt == self.group_range).float()
                group_count = group_map.sum(1)
                group_denom = group_count + (group_count == 0).float()  # avoid nans
                group_loss = (group_map @ loss_per_sample.flatten()) / group_denom

                # update adv_probs
                with torch.no_grad():
                    self.adv_probs = self.adv_probs * torch.exp(
                        args.groupdro_robust_step_size * group_loss.detach()
                    )
                    self.adv_probs = self.adv_probs / (self.adv_probs.sum())

                # compute reweighted robust loss
                loss = group_loss @ self.adv_probs

            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)
            self._scaler_update()
            self.optimizer.zero_grad(set_to_none=True)
            
            accs = meter.compute_accuracy(output, obj_gt)
            losses = meter.compute_loss(output, obj_gt, torch.nn.CrossEntropyLoss(reduction="none"))
            meter.update(losses, accs, obj_gt, group_gt)

            pbar.set_description(
                f"[{self.cur_epoch}/{args.epoch}] loss: {meter.loss_avg.avg():.4f}"
            )
            meter.log_epoch(self.cur_epoch, aligned_topology="firstcol" if args.dataset == "vas" else "diagonal")
            meter.plot_subgroup_metrics(epoch=self.cur_epoch, save_dir="plots", show=False)

        # self.log_to_wandb({"loss": losses.avg})

    def _state_dict_for_save(self):
        state_dict = super()._state_dict_for_save()
        state_dict["adv_probs"] = self.adv_probs
        return state_dict

    def _load_state_dict(self, state_dict):
        super()._load_state_dict(state_dict)
        self.adv_probs = state_dict["adv_probs"]
