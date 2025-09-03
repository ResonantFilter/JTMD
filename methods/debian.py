"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from DebiAN:
# https://github.com/zhihengli-UR/DebiAN
# --------------------------------------------------------

import torch


from tqdm import tqdm
import wandb
from .base_trainer import BaseTrainer
from utils.advanced_metrics import AverageMeter, SubgroupMetricsTracker
from models.classifiers import get_classifier

EPS = 1e-6


class DebiANTrainer(BaseTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "debian"

        default_name = f"{args.method}_es_{args.early_stop_metric}_{args.dataset}"
        self.default_name = default_name

    def _method_specific_setups(self):
        self.second_train_loader = self._get_train_loader(self.train_set)

    def _setup_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")

    def _setup_models(self):
        super()._setup_models()
        args = self.args
        self.bias_discover_net = get_classifier(
            args.arch,
            self.num_classes,
        ).to(self.device)

    def _setup_optimizers(self):
        super()._setup_optimizers()
        args = self.args
        match args.optimizer:
            case "sgd":
                self.optimizer_bias_discover_net = torch.optim.SGD(
                    self.bias_discover_net.parameters(),
                    args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay
                )
            case "adamw":
                self.optimizer_bias_discover_net = torch.optim.AdamW(
                    self.bias_discover_net.parameters(),
                    args.lr,
                    weight_decay=args.weight_decay
                )
            case _:
                raise NotImplementedError()
        

    def _train_classifier(self, images, labels):
        args = self.args
        self.classifier.train()
        self.bias_discover_net.eval()

        image, obj_gt = images, labels[0]
        group_gt = labels[1]

        image = image.to(self.device, non_blocking=True)
        obj_gt = obj_gt.to(self.device, non_blocking=True)
        group_gt = group_gt.to(self.device, non_blocking=True)

        with torch.no_grad():
            spurious_logits = self.bias_discover_net(image)
        with torch.cuda.amp.autocast(enabled=args.amp):
            target_logits = self.classifier(image)

            label = obj_gt.long()
            label = label.reshape(target_logits.shape[0])

            p_vanilla = torch.softmax(target_logits, dim=1)
            p_spurious = torch.sigmoid(spurious_logits)

            ce_loss = self.criterion(target_logits, label)

            # reweight CE with DEO
            for target_val in range(self.num_classes):
                batch_bool = label.long().flatten() == target_val
                if not batch_bool.any():
                    continue
                p_vanilla_w_same_t_val = p_vanilla[batch_bool, target_val]
                p_spurious_w_same_t_val = p_spurious[batch_bool, target_val]

                positive_spurious_group_avg_p = (
                    p_spurious_w_same_t_val * p_vanilla_w_same_t_val
                ).sum() / (p_spurious_w_same_t_val.sum() + EPS)
                negative_spurious_group_avg_p = (
                    (1 - p_spurious_w_same_t_val) * p_vanilla_w_same_t_val
                ).sum() / ((1 - p_spurious_w_same_t_val).sum() + EPS)

                if (
                    negative_spurious_group_avg_p
                    < positive_spurious_group_avg_p
                ):
                    p_spurious_w_same_t_val = 1 - p_spurious_w_same_t_val

                weight = 1 + p_spurious_w_same_t_val
                ce_loss[batch_bool] *= weight

            ce_loss = ce_loss.mean()

        self._loss_backward(ce_loss)
        self._optimizer_step(self.optimizer)
        self.optimizer.zero_grad(set_to_none=True)

        return ce_loss.item(), target_logits

    def _train_bias_discover_net(self, images, labels):
        args = self.args
        self.bias_discover_net.train()
        self.classifier.eval()

        image, obj_gt = images, labels[0]
        group_gt = labels[1]

        image = image.to(self.device, non_blocking=True)
        obj_gt = obj_gt.to(self.device, non_blocking=True)
        group_gt = group_gt.to(self.device, non_blocking=True)

        with torch.no_grad():
            target_logits = self.classifier(image)

        with torch.cuda.amp.autocast(enabled=args.amp):
            spurious_logits = self.bias_discover_net(image)
            label = obj_gt.long()
            label = label.reshape(target_logits.shape[0])
            p_vanilla = torch.softmax(target_logits, dim=1)
            p_spurious = torch.sigmoid(spurious_logits)

            # ==== deo loss ======
            sum_discover_net_deo_loss = 0
            sum_penalty = 0
            num_classes_in_batch = 0
            for target_val in range(self.num_classes):
                batch_bool = label.long().flatten() == target_val
                if not batch_bool.any():
                    continue
                p_vanilla_w_same_t_val = p_vanilla[batch_bool, target_val]
                p_spurious_w_same_t_val = p_spurious[batch_bool, target_val]

                positive_spurious_group_avg_p = (
                    p_spurious_w_same_t_val * p_vanilla_w_same_t_val
                ).sum() / (p_spurious_w_same_t_val.sum() + EPS)
                negative_spurious_group_avg_p = (
                    (1 - p_spurious_w_same_t_val) * p_vanilla_w_same_t_val
                ).sum() / ((1 - p_spurious_w_same_t_val).sum() + EPS)

                discover_net_deo_loss = -torch.log(
                    EPS
                    + torch.abs(
                        positive_spurious_group_avg_p
                        - negative_spurious_group_avg_p
                    )
                )

                negative_p_spurious_w_same_t_val = 1 - p_spurious_w_same_t_val
                penalty = -torch.log(
                    EPS
                    + 1
                    - torch.abs(
                        p_spurious_w_same_t_val.mean()
                        - negative_p_spurious_w_same_t_val.mean()
                    )
                )

                sum_discover_net_deo_loss += discover_net_deo_loss
                sum_penalty += penalty
                num_classes_in_batch += 1

            sum_penalty /= num_classes_in_batch
            sum_discover_net_deo_loss /= num_classes_in_batch
            loss_discover = sum_discover_net_deo_loss + sum_penalty

        self._loss_backward(loss_discover)
        self._optimizer_step(self.optimizer_bias_discover_net)
        self.optimizer_bias_discover_net.zero_grad(set_to_none=True)

        return loss_discover.item()

    def train(self):
        args = self.args
        cls_losses = AverageMeter()
        dis_losses = AverageMeter()
        
        main_meter = SubgroupMetricsTracker(
            num_classes=self.num_classes,
            num_groups=8,
            device=self.device,
            log_history=True,
            wandb_logger = wandb if self.args.wandb else None,
            prefix="tr"
        )

        pbar = tqdm(
            zip(self.train_loader, self.second_train_loader),
            dynamic_ncols=True,
            total=len(self.train_loader),
        )

        for batch, (main_loader, second_loader) in enumerate(pbar):
            main_dat, main_labels, _ = main_loader
            main_dat = main_dat.to(self.device, non_blocking=True)
            main_labels[0] = main_labels[0].to(self.device, non_blocking=True)
            main_labels[1] = main_labels[1].to(self.device, non_blocking=True)
            
            second_dat, second_labels, _ = second_loader
            second_dat = second_dat.to(self.device, non_blocking=True)
            second_labels[0] = second_labels[0].to(self.device, non_blocking=True)
            second_labels[1] = second_labels[1].to(self.device, non_blocking=True)
    
            
            cls_loss, output = self._train_classifier(main_dat, main_labels)
            dis_loss = self._train_bias_discover_net(second_dat, second_labels)

            main_losses = main_meter.compute_loss(output, main_labels[0])
            main_accs   = main_meter.compute_accuracy(output, main_labels[0])

            main_meter.update(main_losses, main_accs, main_labels[0], main_labels[1])
            dis_losses.update(dis_loss, main_labels[0].size(0))            

            self._scaler_update()

            pbar.set_description(
                f"[{self.cur_epoch}/{args.epoch}] cls_loss: {main_meter.loss_avg.avg():.4f} dis_loss: {dis_losses.avg():.4f}"
            )
            main_meter.log_epoch(self.cur_epoch, aligned_topology="firstcol" if args.dataset == "vas" else "diagonal")
            main_meter.plot_subgroup_metrics(epoch=self.cur_epoch, save_dir="plots", show=False)


    def _state_dict_for_save(self):
        state_dict = super()._state_dict_for_save()
        state_dict.update(
            {
                "bias_discover_net": self.bias_discover_net.state_dict(),
                "optimizer_bias_discover_net": self.optimizer_bias_discover_net.state_dict(),
            }
        )
        return state_dict

    def _load_state_dict(self, state_dict):
        super()._load_state_dict(state_dict)
        self.bias_discover_net.load_state_dict(state_dict["bias_discover_net"])
        self.optimizer_bias_discover_net.load_state_dict(
            state_dict["optimizer_bias_discover_net"]
        )
