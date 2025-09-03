import torch

class EMAGPU:
    def __init__(self, label, device, alpha=0.9):
        self.label = label
        self.alpha = alpha
        self.device = device
        self.parameter = torch.zeros(label.size(0), device=device)
        self.updated = torch.zeros(label.size(0), device=device)
        self.num_classes = label.max().item() + 1
        self.max_param_per_class = torch.zeros(self.num_classes, device=device)

    def update(self, data, index):
        self.parameter[index] = (
            self.alpha * self.parameter[index]
            + (1 - self.alpha * self.updated[index]) * data
        )
        self.updated[index] = 1

        # update max_param_per_class
        batch_size = len(index)
        buffer = torch.zeros(batch_size, self.num_classes, device=self.device)
        buffer[range(batch_size), self.label[index]] = self.parameter[index]
        cur_max = buffer.max(dim=0).values
        global_max = torch.maximum(cur_max, self.max_param_per_class)
        label_set_indices = self.label[index].unique()
        self.max_param_per_class[label_set_indices] = global_max[
            label_set_indices
        ]

    def max_loss(self, label):
        return self.max_param_per_class[label]