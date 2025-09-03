#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import numpy as np
import torch
import os
from torchvision import datasets
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from typing import List, Callable, Tuple, Generator, Union
from torch.utils.data import ConcatDataset
import pandas as pd
import gdown
import zipfile

class CMNIST(Dataset):
    DOWNLOAD_URL = "https://drive.google.com/file/d/1QnmRgeuf60vJSM1JU-Rwa3rvKa58IHkx/view?usp=sharing" 
    DATASET_NAME = "cmnist"
    
    base_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    def __init__(
        self, 
        root="./data/cmnist", 
        env="train", 
        bias_amount=0.95, 
        transform=None, 
        return_index = True, 
        external_bias_labels=False
    ):
    
        self.root = root
        if transform is None:
            self.transform = CMNIST.base_transform
        else:
            self.transform = transform
        
        self.env=env
        self.bias_amount=bias_amount
        self.return_index = return_index
        self.num_classes = 10
        self.target_name = "digit"
        self.counfounder_name = "color"
        self.n_confounders = 1
        self.num_groups = 10

        self.bias_folder_dict = {
            95:     "5pct",
            98:     "2pct",
            99:     "1pct",
            99.5: "0.5pct",
        }
        
        if not os.path.isdir(os.path.join(self.root, CMNIST.DATASET_NAME)):
            self.__download_dataset()
        else: self.root = os.path.join(self.root, CMNIST.DATASET_NAME)
        
        if self.env == "train":
            self.filename_array, self.y_array, self.confounder_array = self.load_train_samples()
            if external_bias_labels:
                raise NotImplementedError("Not yet implemented for this dataset")
                print("Loading external bias labels for the training set...")
                self.old_garray = self.confounder_array.copy()
                self.confounder_array = pd.read_csv(os.path.join("outputs", "cmnist_metadata_aug.csv"), header="infer")["ddb"].to_numpy()
                assert len(self.old_garray) == len(self.confounder_array)
                self.old_garray = None

        if self.env == "val":
            self.filename_array, self.y_array, self.confounder_array = self.load_val_samples()

        if self.env == "test":
            self.filename_array, self.y_array, self.confounder_array = self.load_test_samples()
            
        self.group_array = ((self.y_array * (self.num_groups)) + self.confounder_array).long()
        
    def __download_dataset(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        output_path = os.path.join(self.root, "cmnist.zip")
        print(f"=> Downloading {CMNIST.DATASET_NAME} from {CMNIST.DOWNLOAD_URL}")

        try:
            gdown.download(id="1QnmRgeuf60vJSM1JU-Rwa3rvKa58IHkx", output=output_path)
        except:
            raise RuntimeError("Unable to complete dataset download, check for your internet connection or try changing download link.")
        
        print(f"=> Extracting cmnist.zip to directory {self.root}")
        try:
            with zipfile.ZipFile(output_path, mode="r") as unzipper:
                unzipper.extractall(self.root)
        except:
            raise RuntimeError(f"Unable to extract {output_path}, an error occured.")

        self.root = os.path.join(self.root, "cmnist")
        os.remove(output_path)

    def load_train_samples(self):
        samples_path = []
        class_labels=[]
        bias_labels=[]
        bias_folder=self.bias_folder_dict[self.bias_amount]
        for class_folder in sorted(os.listdir(os.path.join(self.root,bias_folder, "align"))):
            for filename in sorted(os.listdir(os.path.join(self.root,bias_folder, "align",class_folder))):
                samples_path.append(os.path.join(self.root,bias_folder, "align",class_folder,filename))
                class_labels.append(self.assign_class_label(filename))
                bias_labels.append(self.assign_bias_label(filename))
        
        for class_folder in sorted(os.listdir(os.path.join(self.root,bias_folder, "conflict"))):
            for filename in sorted(os.listdir(os.path.join(self.root,bias_folder, "conflict",class_folder))):
                samples_path.append(os.path.join(self.root,bias_folder, "conflict",class_folder,filename))
                class_labels.append(self.assign_class_label(filename))
                bias_labels.append(self.assign_bias_label(filename))

        return (
            np.array(samples_path), 
            torch.as_tensor(class_labels), 
            torch.as_tensor(bias_labels)
        ) 
    
    def load_val_samples(self):
        samples_path = []
        class_labels=[]
        bias_labels=[]

        bias_folder=self.bias_folder_dict[self.bias_amount]
        for filename in sorted(os.listdir(os.path.join(self.root,bias_folder,"valid"))):
            samples_path.append(os.path.join(self.root,bias_folder, "valid",filename))
            class_labels.append(self.assign_class_label(filename))
            bias_labels.append(self.assign_bias_label(filename))

        return (
            np.array(samples_path), 
            torch.as_tensor(class_labels), 
            torch.as_tensor(bias_labels)
        ) 
    
    def load_test_samples(self):
        samples_path = []
        class_labels=[]
        bias_labels=[]

        for class_folder in sorted(os.listdir(os.path.join(self.root,"test"))):
            for filename in  sorted(os.listdir(os.path.join(self.root,"test",class_folder))):
                samples_path.append(os.path.join(self.root,"test",class_folder,filename))
                class_labels.append(self.assign_class_label(filename))
                bias_labels.append(self.assign_bias_label(filename))

        return (
            np.array(samples_path), 
            torch.as_tensor(class_labels), 
            torch.as_tensor(bias_labels)
        ) 


    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        file_path = self.filename_array[idx]
        class_label=self.y_array[idx]
        bias_label=self.confounder_array[idx]

        image = self.transform(Image.open(file_path))   #senza self.transofrm per vedere le immagini 
        
        if self.return_index:
            return image, (class_label, bias_label), idx

        return image, class_label, bias_label

    def assign_bias_label(self, filename):
        no_extension=filename.split('.')[0]
        _, y, z = no_extension.split('_')
        y, z = int(y), int(z)
        # if y == z:
        #     return 1
        # return -1
        return z
    
    def assign_class_label(self, filename):
        no_extension=filename.split('.')[0]
        _, y, _ = no_extension.split('_')
        return int(y)
    
    def perclass_populations(self, return_labels: bool = False) -> Union[Tuple[float, float], Tuple[Tuple[float, float], torch.Tensor]]:
        labels: torch.Tensor = torch.zeros(len(self))
        for i in range(len(self)):
            labels[i] = self[i][1][0]

        _, pop_counts = labels.unique(return_counts=True)

        if return_labels:
            return pop_counts.long(), labels.long()

        return pop_counts
    
    def get_sampling_weights(self, classes_only: bool):
        if classes_only:
            group_counts: torch.Tensor = (
                (torch.arange(self.num_classes).unsqueeze(1) == self.y_array)
                .sum(1)
                .float()
            )
        else:
            group_counts: torch.Tensor = (
                (torch.arange(self.num_groups * self.num_classes).unsqueeze(1) == self.group_array)
                .sum(1)
                .float()
            )
        
        group_weights = len(self) / group_counts
        weights = group_weights[self.y_array if classes_only else self.group_array]
        return weights
    
    def set_num_group_and_group_array(self, num_shortcut_category, shortcut_label):
        self.num_groups = self.num_classes * num_shortcut_category
        self.group_array = self.get_labels() * num_shortcut_category + shortcut_label
        
    def set_domain_label(self, shortcut_label):
        self.domain_label = shortcut_label
    
    def get_labels(self) -> torch.Tensor:
        return torch.as_tensor(list([self[i][1][0] for i in range(len(self))]))


    def get_group_labels(self) -> torch.Tensor:
        return torch.as_tensor(list([self[i][1][1] for i in range(len(self))]))
    
    def get_group_array(self):
        return self.group_array
    
    def get_label_array(self):
        return self.y_array
    

    def __repr__(self) -> str:
        return f"CMNIST(env={self.env}, bias_amount={self.bias_amount}, num_classes={self.num_classes})"
     

if __name__ == "__main__":


    train_set=CMNIST(env="train",bias_amount=0.95)
    val_set=CMNIST(env="val",bias_amount=0.95)
    test_set=CMNIST(env="test",bias_amount=0.95)


    #group and display colorized images of the same digit together.
    plt.figure()
    for i in range(0, 55000, 500):
        train_image, l, bl = train_set[i]
        print(train_set.filename_array[i])
        print("class ", l)
        print("bias ", bl)
        plt.imshow(train_image.permute(1,2,0))

        plt.show()

    # for i in range(0, 300, 50):
    #     val_image, l, bl = val_set[i]
    #     print(val_set.samples[i])
    #     print("class ", l)
    #     print("bias ", bl)
    #     plt.imshow(val_image)

    #     plt.show()

    # for i in range(0, 4000, 100):
    #     test_image, l, bl = test_set[i]
    #     print(test_set.samples[i])
    #     print("class ", l)
    #     print("bias ", bl)
    #     plt.imshow(test_image)

    #     plt.show()
