import torch
import pandas as pd
import os
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += value * n
        self.count += n

    def avg(self):
        return (self.total / self.count) if self.count > 0 else 0.0
    
    def __repr__(self) -> str:
        return f"AverageMeter(avg={self.avg()}, nvalues={self.count})"
    
def eval_inacc_and_gap(train_set, meter: "SubgroupMetricsTracker"):    
    _, group_counts = train_set.get_sampling_weights(classes_only=False).unique(return_counts=True)    
    train_distribution = (group_counts / len(train_set)).to(meter.device)
    per_class_avg = meter.acc_meter.per_subgroup_avg().mean(dim=0)
    in_acc = torch.sum(per_class_avg * train_distribution).to(meter.device)
    gaps = torch.ones(len(per_class_avg)).to(meter.device) * in_acc - per_class_avg
    gaps[0] = in_acc
    return gaps

class _ScalarSubgroupAverageMeter:
    def __init__(self, num_classes, num_groups, device="cpu"):
        self.num_classes = num_classes
        self.num_groups = num_groups
        self.device = torch.device(device)
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0
        self.sum_matrix = torch.zeros(self.num_classes, self.num_groups, device=self.device)
        self.count_matrix = torch.zeros(self.num_classes, self.num_groups, device=self.device)

    @torch.no_grad()
    def update(self, values, class_labels, group_labels):
        values = values.to(self.device)
        class_labels = class_labels.to(self.device)
        group_labels = group_labels.to(self.device)

        self.total += values.sum().item()
        self.count += values.numel()

        for c in range(self.num_classes):
            for g in range(self.num_groups):
                mask = (class_labels == c) & (group_labels == g)
                if mask.any():
                    self.sum_matrix[c, g] += values[mask].sum()
                    self.count_matrix[c, g] += mask.sum()

    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0.0

    @torch.no_grad()
    def per_subgroup_avg(self):
        with torch.no_grad():
            avg_matrix = torch.zeros_like(self.sum_matrix)
            nonzero_mask = self.count_matrix > 0
            avg_matrix[nonzero_mask] = self.sum_matrix[nonzero_mask] / self.count_matrix[nonzero_mask]
        return avg_matrix

    def to_dataframe(self, metric_name, epoch=None):
        avg_matrix = self.per_subgroup_avg().cpu()
        records = []
        for c in range(self.num_classes):
            for g in range(self.num_groups):
                value = avg_matrix[c, g].item()
                entry = {'epoch': epoch, 'class': c, 'group': g, metric_name: value}
                records.append(entry)
        return pd.DataFrame(records)


class SubgroupMetricsTracker:
    def __init__(self, num_classes, num_groups, device="cpu", log_history=True, wandb_logger=None, prefix=""):
        self.device = device
        self.num_classes = num_classes
        self.num_groups = num_groups
        self.log_history = log_history
        self.wandb_logger = wandb_logger
        self.prefix = prefix

        self.loss_meter = _ScalarSubgroupAverageMeter(num_classes, num_groups, device)
        self.acc_meter = _ScalarSubgroupAverageMeter(num_classes, num_groups, device)
        self.loss_avg = AverageMeter()
        self.acc_avg = AverageMeter()
        self.history = []  # contains list of dicts (epoch, loss, acc)

    def reset(self):
        self.loss_meter.reset()
        self.acc_meter.reset()
        self.loss_avg.reset()
        self.acc_avg.reset()

    @torch.no_grad()
    def update(self, loss_values, acc_values, class_labels, group_labels):
        # print("acc values:", acc_values)
        # print("acc sum:", acc_values.sum().item())
        # print("acc numel:", acc_values.numel())
        # print("acc avg (should be ≤ 1.0):", acc_values.sum().item() / acc_values.numel())

        self.loss_meter.update(loss_values, class_labels, group_labels)
        self.acc_meter.update(acc_values, class_labels, group_labels)
        self.loss_avg.update(loss_values.mean().item(), loss_values.numel())
        self.acc_avg.update(acc_values.mean().item(), acc_values.numel())

    def log_epoch(self, epoch, aligned_topology="diagonal"):
        global_loss = self.loss_avg.avg()
        global_acc = self.acc_avg.avg()
        # subgroup_loss_df = self.loss_meter.to_dataframe("loss", epoch)
        # subgroup_acc_df = self.acc_meter.to_dataframe("accuracy", epoch)
        acc_matrix =  self.acc_meter.per_subgroup_avg()
        
        average_acc = self.acc_meter.global_avg()
        
        match aligned_topology:
            case "diagonal":
                align_acc = self.acc_meter.sum_matrix.diag().sum() / self.acc_meter.count_matrix.diag().sum()
                temp_sum_matrix = self.acc_meter.sum_matrix.clone()
                temp_cnt_matrix = self.acc_meter.count_matrix.clone()
                temp_sum_matrix.diagonal().zero_()
                temp_cnt_matrix.diagonal().zero_()                
                confl_acc = temp_sum_matrix[temp_cnt_matrix != 0].sum() / temp_cnt_matrix[temp_cnt_matrix != 0].sum()
                
            case "firstcol":
                align_acc = self.acc_meter.sum_matrix[:, 0].sum() / self.acc_meter.count_matrix[:, 0].sum()
                confl_acc = self.acc_meter.sum_matrix[:, 1:].sum() / self.acc_meter.count_matrix[:, 1:].sum()
            
            case _:
                raise ValueError("Unrecognized 'aligned_topology'. Use either 'diagonal' or 'firstcol'.")
                

        if self.log_history:
            self.history.append({
                f"{self.prefix}_epoch": epoch,
                f"{self.prefix}_global_loss": global_loss,
                f"{self.prefix}_global_acc":  global_acc,
                f"{self.prefix}_align_acc": align_acc,
                f"{self.prefix}_confl_acc": confl_acc
            })

        if self.wandb_logger is not None:
            print(self.wandb_logger)
            self.wandb_logger.log({
                f"{self.prefix}_epoch": epoch,
                f"{self.prefix}_global_loss": global_loss,
                f"{self.prefix}_global_acc":  global_acc,
                f"{self.prefix}_align_acc": align_acc,
                f"{self.prefix}_confl_acc": confl_acc
            }, step=epoch)
            # self.wandb_logger.log({
            #     f"{self.prefix}_subgroup_loss": wandb.Table(dataframe=subgroup_loss_df),
            #     f"{self.prefix}_subgroup_accuracy": wandb.Table(dataframe=subgroup_acc_df)
            # }, step=epoch)
            
            acc_to_log = {}
            loss_to_log = {}
            for c in range(self.acc_meter.num_classes):
                for g in range(self.acc_meter.num_groups):
                    acc_to_log[f"{self.prefix}-cl_{c}-gr_{g}"] = acc_matrix[c, g]
                    loss_to_log[f"{self.prefix}-cl_{c}-gr_{g}"] = acc_matrix[c, g]
                    
            self.wandb_logger.log(acc_to_log, step=epoch)
            self.wandb_logger.log(loss_to_log, step=epoch)

    def print_summary(self):
        print(f"Global Loss: {self.loss_avg.avg():.4f}")
        print(f"Global Acc : { self.acc_avg.avg():.4f}")
        print("\nSubgroup Loss Matrix:")
        print(self.loss_meter.per_subgroup_avg())
        print("\nSubgroup Accuracy Matrix:")
        print(self.acc_meter.per_subgroup_avg())
        
    @torch.no_grad()
    def compute_accuracy(self, outputs, targets, threshold=0.5):
        """
        Supports both binary and multiclass classification.
        outputs: logits or probabilities of shape [B, C] or [B]
        targets: labels of shape [B] (long or float)
        Returns: float tensor of shape [B] with accuracy per sample (0 or 1)
        """
        targets = targets.long()
        if outputs.ndim == 1 or outputs.shape[1] == 1:
            # Binary classification (logits or probs)
            preds = (torch.sigmoid(outputs.squeeze()) > threshold).long()
            acc = (preds == targets.long()).float() 
        else:
            # Multiclass classification
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == targets.long()).float() 
        return acc

    @torch.no_grad()
    def compute_loss(self, outputs, targets, loss_fn=torch.nn.CrossEntropyLoss(reduction="none")):
        """
        Computes per-sample loss.
        Supports any reduction='none' compatible loss function.
        outputs: model predictions [B, C] or [B]
        targets: labels [B]
        loss_fn: e.g. nn.CrossEntropyLoss(reduction='none') or BCEWithLogitsLoss
        Returns: Tensor [B] of scalar losses
        """
        return loss_fn(outputs, targets)
    
    def tqdm_postfix(self, digits=2, class_avg=True, worst_group=True, limit=12):
        """
        Returns a dictionary for tqdm live updates.
        - digits: decimal precision
        - class_avg: include per-class averages (first few)
        - worst_group: include worst-case subgroup metrics
        - limit: number of classes to show individually
        """
        summary = {
            f"{self.prefix}_loss": f"{self.loss_avg.avg():.{digits}f}",
            f"{self.prefix}_acc": f"{ self.acc_avg.avg():.{digits}f}"
        }

        loss_matrix = self.loss_meter.per_subgroup_avg()
        acc_matrix = self.acc_meter.per_subgroup_avg()

        if class_avg:
            class_loss_avg = loss_matrix.mean(dim=1)
            class_acc_avg = acc_matrix.mean(dim=1)
            for c in range(min(limit, self.num_classes)):
                # summary[f"{self.prefix}_cls{c}_l"] = f"{class_loss_avg[c].item():.{digits}f}"
                summary[f"{self.prefix}_cls{c}_a"] = f"{class_acc_avg[c].item():.{digits}f}"

        if worst_group:
            worst_loss, _ = loss_matrix.max(dim=1)
            worst_acc, _ = acc_matrix.min(dim=1)
            for c in range(min(limit, self.num_classes)):
                if self.acc_meter.count_matrix[c].sum() == 0:
                    continue
                # summary[f"{self.prefix}_wcls{c}_l"] = f"{worst_loss[c].item():.{digits}f}"
                summary[f"{self.prefix}_wcls{c}_a"] = f"{worst_acc[c].item():.{digits}f}"

        return summary
    
    def plot_subgroup_metrics(self, epoch=None, save_dir=None, show=False, annotate_heatmaps=True, fontsize=24, figsize=(24, 16), show_gap=False):
        loss_matrix = self.loss_meter.per_subgroup_avg()
        acc_matrix =  self.acc_meter.per_subgroup_avg()

        pergroup_avg = acc_matrix.mean(dim=0).cpu()
        gap_wrt_first = pergroup_avg - pergroup_avg[0]

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, height_ratios=[4, 1.2], hspace=0.25)


        loss_matrix = loss_matrix.cpu()
        acc_matrix  = acc_matrix.cpu()
        # Loss heatmap (top left)
        ax_loss = fig.add_subplot(gs[0, 0])
        sns.heatmap(
            loss_matrix,
            ax=ax_loss,
            annot=annotate_heatmaps,
            fmt=".2f",
            cmap="Reds",
            cbar=True,
            annot_kws={"size": fontsize},
        )
        ax_loss.set_title("Subgroup Loss Heatmap")
        ax_loss.set_xlabel("Group")
        ax_loss.set_ylabel("Class")

        # Accuracy heatmap (top right)
        ax_acc = fig.add_subplot(gs[0, 1])
        sns.heatmap(
            acc_matrix,
            ax=ax_acc,
            annot=annotate_heatmaps,
            fmt=".2f",
            cmap="Blues",
            cbar=True,
            annot_kws={"size": fontsize},
            vmin=0.0,
            vmax=1.0
        )
        ax_acc.set_title("Subgroup Accuracy Heatmap")
        ax_acc.set_xlabel("Group")
        ax_acc.set_ylabel("Class")

        if show_gap:
            # Per-group metrics heatmap (bottom right)
            ax_avg = fig.add_subplot(gs[1, 1])
            arr = torch.vstack([pergroup_avg, gap_wrt_first]).numpy()
            sns.heatmap(
                arr,
                ax=ax_avg,
                annot=True,
                fmt=".2f",
                cmap="Purples",
                cbar=True,
                yticklabels=["Group avg", "Gap wrt group 0"],
                xticklabels=[str(i) for i in range(acc_matrix.shape[1])],
                annot_kws={"size": fontsize},
                vmin=-1.0,
                vmax=+1.0
                
            )
            ax_avg.set_xlabel("Group")
            ax_avg.set_title("Per-group avg & gap (accuracy)")
        else:
            ax_unused = fig.add_subplot(gs[1, 1])
            ax_unused.axis("off")

        # Hide bottom left subplot (not used)
        ax_unused = fig.add_subplot(gs[1, 0])
        ax_unused.axis("off")

        # Main title and tight layout
        fig.suptitle(f"{self.prefix}, epoch: {epoch}", fontsize=fontsize+4)        

        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            suffix = f"epoch{epoch}" if epoch is not None else "current"
            out_path = os.path.join(save_dir, f"{self.prefix}_subgroup_metrics_{suffix}.png")
            plt.savefig(out_path)

        if self.wandb_logger:
            import wandb
            self.wandb_logger.log({
                f"{self.prefix}_subgroup_metrics/global": wandb.Image(plt)
            }, step=epoch)
                

        if show:
            plt.show()
        else:
            plt.close(fig)

    def export_csv(self, folder_path="logs", prefix="metrics"):
        os.makedirs(folder_path, exist_ok=True)

        history_df = pd.DataFrame(self.history)
        history_df.to_csv(os.path.join(folder_path, f"{self.prefix}_{prefix}_history.csv"), index=False)

        all_loss_dfs = []
        all_acc_dfs = []

        for record in self.history:
            epoch = record["{self.prefix}_epoch"]
            loss_df = self.loss_meter.to_dataframe("loss", epoch)
            acc_df = self.acc_meter.to_dataframe("accuracy", epoch)
            all_loss_dfs.append(loss_df)
            all_acc_dfs.append(acc_df)

        if all_loss_dfs:
            pd.concat(all_loss_dfs).to_csv(os.path.join(folder_path, f"{self.prefix}_{prefix}_subgroup_loss.csv"), index=False)
        if all_acc_dfs:
            pd.concat(all_acc_dfs).to_csv(os.path.join(folder_path, f"{self.prefix}_{prefix}_subgroup_accuracy.csv"), index=False)
