"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch

import wandb
from tqdm import tqdm
from .base_trainer import BaseTrainer
from ..utils.advanced_metrics import AverageMeter, SubgroupMetricsTracker


class ERMTrainer(BaseTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "erm"
        default_name = f"{args.method}_{args.dataset}"
        self.default_name = default_name

    def train(self):
        args = self.args
        self._set_train()
        samples_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        losses = AverageMeter()
        meter = SubgroupMetricsTracker(
            num_classes=self.num_classes,
            num_groups=self.num_groups,
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
            image = image.to(self.device)
            obj_gt = obj_gt.to(self.device)
            group_gt = group_gt.to(self.device)

            with torch.amp.autocast("cuda", enabled=args.amp):
                output = self.classifier(image)
                loss = self.criterion(output, obj_gt)

            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)
            self._scaler_update()
            self.optimizer.zero_grad(set_to_none=True)

            accs = meter.compute_accuracy(output, obj_gt)
            losses = meter.compute_loss(output, obj_gt, samples_loss_fn)
            meter.update(losses, accs, obj_gt, group_gt)

            pbar.set_description(
                f"[{self.cur_epoch}/{args.epoch}] loss: {meter.loss_avg.avg():.4f}"
            )
            meter.log_epoch(self.cur_epoch, aligned_topology="firstcol" if args.dataset == "vas" else "diagonal")
            # meter.plot_subgroup_metrics(epoch=self.cur_epoch, save_dir="plots", show=False)