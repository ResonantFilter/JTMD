
import torch
import torchvision
import pandas as pd
import os
from tqdm import tqdm
from utils.advanced_metrics import *
import itertools

PATH_TO_MODELS = "saved_models"


def train_model_erm(model: torch.nn.Module, train_loader, val_loader, test_loader, device, optimizer: torch.optim.Optimizer, num_classes, num_groups = 16, epochs=10, wb=None, make_figures=False, name="jtt"):
    cur_model_name = f"{name}-init.pt"
    print(f"Saving {cur_model_name}")
    torch.save(model.state_dict(), os.path.join(PATH_TO_MODELS,cur_model_name))
    
    subgroup_meter = SubgroupMetricsTracker(
        num_classes=num_classes,
        num_groups=num_groups,
        device="cuda",
        log_history=True,
        wandb_logger=wb,
        prefix="tr"
    )
    
    for epoch in range(epochs):
        subgroup_meter.reset()
        
        tk0 = tqdm(
            train_loader, total=int(len(train_loader)), leave=True, dynamic_ncols=True
        )
        model.train()
        
        with torch.enable_grad():
            for batch, (dat, labels, _) in enumerate(tk0):
                dat = dat.to(device)
                target = labels[0].to(device).long()
                bias_l = labels[1].to(device).long()
                output = model(dat)
                if len(labels) > 2:
                    labels = torch.vstack(labels)
            
                loss_task: torch.Tensor = model.loss_fn(output, target)
                loss_task.backward()
                optimizer.step()
                optimizer.zero_grad()

                accs = subgroup_meter.compute_accuracy(output, target)
                losses = subgroup_meter.compute_loss(output, target, torch.nn.CrossEntropyLoss(reduction="none"))        
                subgroup_meter.update(losses, accs, target, bias_l)
                tk0.set_postfix(subgroup_meter.tqdm_postfix(class_avg=False))             
                torch.save(model.state_dict(), os.path.join(PATH_TO_MODELS, cur_model_name))
        
        # if wb is not None:
        #     wb.log_output(postifix_dict)
        
        subgroup_meter.log_epoch(self.cur_epoch, aligned_topology="firstcol" if args.dataset == "vas" else "diagonal")
        subgroup_meter.plot_subgroup_metrics(epoch=epoch, save_dir="plots", show=False)

        if val_loader is not None:
            evaluate_model(
                model, 
                val_loader, 
                num_classes, 
                num_groups=num_groups, 
                criterion=torch.nn.CrossEntropyLoss(),
                epoch=epoch,
                device=device,
                wb=wb,
                prefix="val"
            )

        if test_loader is not None:        
            evaluate_model(
                model, 
                test_loader, 
                num_classes, 
                num_groups=num_groups, 
                criterion=torch.nn.CrossEntropyLoss(),
                epoch=epoch,
                device=device,
                wb=wb,
                prefix="te"
            )        

    cur_model_name = f"{name}-final.pt"
    print(f"Saving {cur_model_name}")
    torch.save(model.state_dict(), os.path.join(PATH_TO_MODELS, cur_model_name))
    return cur_model_name

@torch.no_grad()
def evaluate_model(model, test_loader, num_classes, num_groups, criterion, epoch, device, wb, prefix="val", config=None):
    model.eval()
    subgroup_top1 = SubgroupMetricsTracker(
        num_classes=num_classes,
        num_groups=num_groups,
        device="cuda",
        log_history=True,
        wandb_logger=wb,
        prefix=prefix,
    )
    
    if config is not None:
        match config.dataset:
            case "vas":
                aligned_topology = "firstcol"
            case _:
                aligned_topology = "diagonal"
    
    tk0 = tqdm(
        test_loader, total=int(len(test_loader)), leave=True, dynamic_ncols=True
    )
    
    for batch, (dat, labels, _) in enumerate(tk0):
        dat     : torch.Tensor = dat.to(device)
        target  : torch.Tensor = labels[0].to(device)
        bias_l  : torch.Tensor = labels[1].to(device)
        output  : torch.Tensor = model(dat)
        
        accs = subgroup_top1.compute_accuracy(output, target)
        losses = subgroup_top1.compute_loss(output, target, torch.nn.CrossEntropyLoss(reduction="none"))        
        subgroup_top1.update(losses, accs, target, bias_l)
        tk0.set_postfix(subgroup_top1.tqdm_postfix(class_avg=False))
    
    if wb is not None:    
        subgroup_top1.log_epoch(epoch, aligned_topology=aligned_topology)
        subgroup_top1.plot_subgroup_metrics(epoch=epoch, save_dir="plots", show=False)

    return subgroup_top1.loss_avg.avg(), subgroup_top1
