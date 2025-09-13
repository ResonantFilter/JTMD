import torch

import wandb
from tqdm import tqdm
from .base_trainer import BaseTrainer
from ..utils.advanced_metrics import AverageMeter, SubgroupMetricsTracker


class NaiveSupervisedTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "naivesupervised"
        default_name = f"{args.method}_{args.dataset}"
        self.default_name = default_name
        args.reweight_groups = True
        self.uw_factor = args.uw_factor
        self.dw_factor = args.dw_factor

    def _setup_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")

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
                loss[group_gt != obj_gt] *= self.uw_factor
                loss[group_gt == obj_gt] *= self.dw_factor

                loss = loss.mean()

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