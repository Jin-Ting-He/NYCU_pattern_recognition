import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from efficientnet_pytorch import EfficientNet


class BagClassifier(nn.Module):
    def __init__(self):
        super(BagClassifier, self).__init__()
        self.batch_size = 1
        # weights_url = 'https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth'
        # torch.hub.download_url_to_file(weights_url, 'efficientnet_b0.pth')
        # self.base_model = models.efficientnet_b0()
        # state_dict = torch.load('efficientnet_b0.pth')
        # self.base_model.load_state_dict(state_dict)
        # self.base_model.features = nn.Sequential(*list(self.base_model.features.children())[:-1])

        # # Freeze EfficientNet weights
        # for param in self.base_model.parameters():
        #     param.requires_grad = False
        self.model = EfficientNet.from_pretrained('efficientnet-b7')
        for param in self.model.parameters():
            param.requires_grad = False
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2560, 1)
    
    def forward(self, x):
        batch_size, num_instances, c, h, w = x.size()
        x = x.view(-1, c, h, w)  # Reshape to (batch_size * num_instances, c, h, w)
        features = self.model.extract_features(x)
        x = self.avg_pool(features)
        x = x.view(x.size(0), -1)
        x = x.view(batch_size, num_instances, -1)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x
        # Process images in batches
        # features = []
        # for i in range(0, x.size(0), self.batch_size):
        #     batch = x[i:i + self.batch_size]
        #     batch_features = self.model.extract_features(batch)
        #     batch_features = self.avg_pool(batch_features)
        #     batch_features = batch_features.view(batch_features.size(0), -1)
        #     features.append(batch_features)
        #     del batch_features
        #     torch.cuda.empty_cache()

        # features = torch.cat(features, dim=0)
        # features = features.view(batch_size, num_instances, -1)
        # features = torch.mean(features, dim=1)
        # output = self.fc(features)
        # return output


        # x = self.base_model.features(x)
        # print(x.shape)
        # x = self.avg_pool(features)
        # # print(x.shape)
        # x = x.view(x.size(0), -1)
        # x = x.view(batch_size, num_instances, -1)
        # x = torch.mean(x, dim=1)
        # x = self.fc(x)
        # return x