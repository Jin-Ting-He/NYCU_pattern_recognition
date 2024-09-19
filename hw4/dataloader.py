import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.labels = []
        self.file_name = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if data_dir != 'released/test':
            self.mode = "train"
            for label, class_dir in enumerate(['class_0', 'class_1']):
                class_path = os.path.join(data_dir, class_dir)
                for file_name in os.listdir(class_path):
                    file_path = os.path.join(class_path, file_name)
                    self.data.append(file_path)
                    self.labels.append(label)
        else:
            self.mode = "test"
            for file_name in os.listdir(data_dir):
                file_path = os.path.join(data_dir, file_name)
                self.data.append(file_path)
                self.file_name.append(file_name.split('.')[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        with open(file_path, 'rb') as file:
            sample = pickle.load(file)
        
        if self.mode == 'train':
            label = self.labels[idx]
            if self.transform:
                sample = np.array([self.transform(Image.fromarray(img)) for img in sample])
            return torch.tensor(sample), torch.tensor(label)
        else:
            if self.transform:
                sample = np.array([self.transform(Image.fromarray(img)) for img in sample])
            return torch.tensor(sample), self.file_name[idx]