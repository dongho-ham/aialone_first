# 설계한 모델을 저장하는 파일입니다.

import torch.nn as nn
import torch


class myLeNet(nn.Module):
    def __init__(self, num_classes): 
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5) 
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.act1 = nn.ReLU() 

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) 
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.act2 = nn.ReLU() 

        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x): 
        b, _, _, _ = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x) 
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x) 
        x = self.pool2(x)

        x = x.reshape(b, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x 

class myLeNet_seq(nn.Module):
    def __init__(self, num_classes): 
        super().__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5) ,
            nn.BatchNorm2d(num_features=6),
            nn.ReLU() ,

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) ,
            nn.BatchNorm2d(num_features=16),
            nn.ReLU() ,

            nn.MaxPool2d(kernel_size=2),
        )
        self.seq2 = nn.Sequential(
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Linear(in_features=120, out_features=84),
            nn.Linear(in_features=84, out_features=num_classes),
        ) 
        

    def forward(self, x): 
        b, c, h, w = x.shape
        x = self.seq1(x)
        x = x.reshape(b, -1)
        x = self.seq2(x)
        return x 

class myLeNet_linear(nn.Module):
    def __init__(self, num_classes): 
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5) 
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.act1 = nn.ReLU() 

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.fc_1 = nn.Linear(6*14*14, 2048)
        self.fc_2 = nn.Linear(2048, 6*14*14)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) 
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.act2 = nn.ReLU() 

        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x): 
        b, c, h, w = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x) 
        x = self.pool1(x)

        _, tmp_c, tmp_w, tmp_h = x.shape
        x = x.reshape(b, -1) 
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = x.reshape(b, tmp_c, tmp_w, tmp_h) 

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x) 
        x = self.pool2(x)

        x = x.reshape(b, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x 

class myLeNet_convs(nn.Module):
    def __init__(self, num_classes): 
        super().__init__()
        self.tmp_conv1 = nn.ModuleList(
            [nn.Conv2d(3, 6, 3, 1, 1)] + [
            nn.Conv2d(6, 6, 3, 1, 1) for _ in range(3-1)
        ])
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=5) 
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.act1 = nn.ReLU() 

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) 
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.act2 = nn.ReLU() 

        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x): 
        b, c, h, w = x.shape
        for module in self.tmp_conv1: 
            x = module(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x) 
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x) 
        x = self.pool2(x)

        x = x.reshape(b, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x 

class myLeNet_incep(nn.Module):
        def __init__(self, num_classes): 
            super().__init__()
            self.conv1_1 = nn.Conv2d(3, 6, 5, 1, 2)
            self.conv1_2 = nn.Conv2d(3, 6, 3, 1, 1)
            self.conv1_3 = nn.Conv2d(3, 6, 1, 1, 0)

            self.conv1 = nn.Conv2d(in_channels=18, out_channels=6, kernel_size=5) 
            self.bn1 = nn.BatchNorm2d(num_features=6)
            self.act1 = nn.ReLU() 

            self.pool1 = nn.MaxPool2d(kernel_size=2)

            self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) 
            self.bn2 = nn.BatchNorm2d(num_features=16)
            self.act2 = nn.ReLU() 

            self.pool2 = nn.MaxPool2d(kernel_size=2)
            
            self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
            self.fc2 = nn.Linear(in_features=120, out_features=84)
            self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

        def forward(self, x): 
            b, c, h, w = x.shape
            x_1 = self.conv1_1(x)
            x_2 = self.conv1_2(x)
            x_3 = self.conv1_3(x)

            x_cat = torch.cat((x_1, x_2, x_3), dim=1)
            x = self.conv1(x_cat)
            x = self.bn1(x)
            x = self.act1(x) 
            x = self.pool1(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.act2(x) 
            x = self.pool2(x)

            x = x.reshape(b, -1)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x
