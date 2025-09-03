"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from .erm import ERMTrainer
from .lff import LfFTrainer
from .groupdro import GroupDROTrainer
from .eiil import EIILTrainer
from .jtt import JTTTrainer
from .debian import DebiANTrainer


method_to_trainer = {
    "erm": ERMTrainer,
    "lff": LfFTrainer,
    "groupdro": GroupDROTrainer,
    "eiil": EIILTrainer,
    "jtt": JTTTrainer,
    "debian": DebiANTrainer,
}
