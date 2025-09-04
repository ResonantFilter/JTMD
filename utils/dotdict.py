import torch
from torch.utils import data
from torchvision import transforms
import os

class DotDict(dict):
    """
    A dictionary that allows accessing entries with dot notation.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

# Example usage:
# config_dict = {'learning_rate': 0.001, 'batch_size': 32}
# config = DotDict(config_dict)
# print(config.learning_rate)
# config.epochs = 10
# print(config['epochs'])