"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm


def get_classifier(arch, num_classes, weights="IMAGENET1K_V1", new_fc=True):
    if arch.startswith("resnet"):
        model = models.__dict__[arch](weights=weights)
        if new_fc:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch.startswith("mlp"):
        if weights != "none":
            print("WARNING! ImageNet pretraining ignored for MLP models")
        model = CMNISTMLP()
    else:
        raise NotImplementedError

    return model


def get_transforms(arch, is_training):
    if arch.startswith("resnet"):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if is_training:
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
    elif arch.startswith("mlp"):
        transform = transforms.ToTensor()
    else:
        raise NotImplementedError

    return transform


class CMNISTMLP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(3*28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 16),
            nn.ReLU()
        )
        self.fc = nn.Linear(16, 10)
        
    def forward(self, x: torch.Tensor):
        x = x.flatten(start_dim=1)
        x = self.feature_extractor(x)
        return self.fc(x)
    


class DomainIndependentClassifier(nn.Module):
    def __init__(self, arch, num_classes, num_domain, weights="IMAGENET1K_V1"):
        super(DomainIndependentClassifier, self).__init__()
        self.backbone = get_classifier(arch, num_classes, weights=weights)
        self.domain_classifier_list = nn.ModuleList(
            [
                nn.Linear(self.backbone.fc.in_features, num_classes)
                for _ in range(num_domain)
            ]
        )
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        logits_per_domain = [
            classifier(x) for classifier in self.domain_classifier_list
        ]
        logits_per_domain = torch.stack(logits_per_domain, dim=1)

        if self.training:
            return logits_per_domain
        else:
            return logits_per_domain.mean(dim=1)


class LastLayerEnsemble(nn.Module):
    def __init__(
        self,
        num_classes,
        num_dist_shift,
        backbone=None,
        in_features=None,
    ) -> None:
        super().__init__()
        assert not (backbone is not None and in_features is not None)
        if backbone is not None:
            self.backbone = backbone
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif in_features is not None:
            self.backbone = None
        else:
            raise ValueError("Either backbone or in_features must be provided")

        self.ensemble_classifier_list = nn.ModuleList(
            [nn.Linear(in_features, num_classes) for _ in range(num_dist_shift)]
        )

        self.dist_shift_predictor = nn.Linear(in_features, num_dist_shift)

    def forward(self, x):
        if self.backbone is not None:
            x = self.backbone(x)

        logits_per_dist_shift = [
            classifier(x) for classifier in self.ensemble_classifier_list
        ]
        logits_per_dist_shift = torch.stack(logits_per_dist_shift, dim=1)

        if self.training:
            dist_shift = self.dist_shift_predictor(x)
            return logits_per_dist_shift, dist_shift
        else:
            dist_shift = F.softmax(self.dist_shift_predictor(x), dim=1)
            return (logits_per_dist_shift * dist_shift.unsqueeze(-1)).sum(dim=1)
