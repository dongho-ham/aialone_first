import torch.nn as nn
import torch

num_blocks_18 = [2, 2, 2, 2]
num_blocks_34 = [3, 4, 6, 3]
num_blocks_50 = [3, 4, 6, 3]
num_blocks_101 = [3, 4, 23, 3]
num_blocks_152 = [3, 8, 36, 3]
num_channel_33 = [64, 64, 128, 256, 512]
num_channel_131 = [64, 256, 512, 1024, 2048]

class ResNet_front(nn.Module):
    def __init__(self): 
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(3, 2, 1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x 

class ResNet_back(nn.Module):
    def __init__(self, num_classes = 10, config = '18'):
        super().__init__()
        in_feat = 512 if config in ['18', '34'] else 2048
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_feat, num_classes)
    
    def forward(self, x):
        x = self.pool(x)
        # 차원이 [100, 512, 1, 1]이기 때문에 squeeze를 사용해 1을 제거
        x = torch.squeeze(x)
        x = self.fc(x)
        return x

class ResNet_block(nn.Module):
    def __init__(self, in_channel, out_channel, downsampling=False):
        super().__init__()
        self.downsampling = downsampling
        stride = 1
        if self.downsampling:
            stride = 2
            self.skip_conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride, 1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
            )
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
            )
        self.second_conv = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        skip_x = torch.clone(x)
        
        if self.downsampling:
            skip_x = self.skip_conv(skip_x)

        x = self.first_conv(x)
        x = self.second_conv(x)

        x = x+skip_x
        x = self.relu(x)
        return x
    
class ResNet_bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, downsampling=False):
        super().__init__()
        stride = 2 if downsampling else 1

        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel // 4, 1, stride),
            nn.BatchNorm2d(out_channel // 4),
            nn.ReLU()
            )
        self.second_conv = nn.Sequential(
            nn.Conv2d(out_channel // 4, out_channel // 4, 3, 1, 1),
            nn.BatchNorm2d(out_channel // 4),
            nn.ReLU()
        )
        self.third_conv = nn.Sequential(
            nn.Conv2d(out_channel // 4, out_channel, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.relu = nn.ReLU()
    def forward(self, x):
        skip_x = torch.clone(x)
        
        skip_x = self.skip_conv(skip_x)

        x = self.first_conv(x)
        x = self.second_conv(x)
        x = self.third_conv(x)

        x = x+skip_x
        x = self.relu(x)
        return x

class ResNet_middle(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config == '18':
            num_blocks, num_channel = num_blocks_18, num_channel_33
            self.target_layer = ResNet_block
        if config == '34':
            num_blocks, num_channel = num_blocks_34, num_channel_33
            self.target_layer = ResNet_block
        if config == '50':
            num_blocks, num_channel = num_blocks_50, num_channel_131
            self.target_layer = ResNet_bottleneck
        if config == '101':
            num_blocks, num_channel = num_blocks_101, num_channel_131
            self.target_layer = ResNet_bottleneck
        if config == '151':
            num_blocks, num_channel = num_blocks_152, num_channel_131
            self.target_layer = ResNet_bottleneck

        # NUM_CHANNEL_131 = [64, 256, 512, 1024, 2048]
        self.layer1 = self.make_layer(num_channel[0], num_channel[1], num_blocks[0])
        self.layer2 = self.make_layer(num_channel[1] // 2, num_channel[2], num_blocks[1], True)
        self.layer3 = self.make_layer(num_channel[2] // 2, num_channel[3], num_blocks[2], True)
        self.layer4 = self.make_layer(num_channel[3] // 2, num_channel[4], num_blocks[3], True)
        
    def make_layer(self, in_channel, out_channel, num_blocks, downsampling=False):
        layer = [self.target_layer(in_channel, out_channel, downsampling)]
        for _ in range(num_blocks-1):
            layer.append(self.target_layer(out_channel, out_channel, downsampling))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.res_front = ResNet_front()
        self.res_mid = ResNet_middle(args.res_config)
        self.res_back = ResNet_back(args.num_classes, args.res_config)

    def forward(self, x):
        x = self.res_front(x)
        x = self.res_mid(x)
        x = self.res_back(x)

        return x
