import torch.nn as nn
import torch

# element wise를 위한 conv layer 추가
# 18을 기준으로 구현 완료

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
    def __init__(self, num_classes=10, config='18'):
        super().__init__()
        # adaptive ap 추가 공부
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_feat = 512 if config in ['18', '34'] else 2048
        self.fc = nn.Linear(in_feat, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.fc(x)
        return x
    
class resnet_block(nn.Moduel):
    def __init__(self, in_channel, out_channel, downsampling=False):
        super().__init__()
        # forward에서 사용하기 위해 객체 변수로 만듦
        self.downsampling = downsampling
        stride = 1
        if downsampling:
            stride=2
            self.skip_conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride, 1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
            )
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.second_conv = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        skip_x = torch.clone(x)
        # element wise를 위한 downsampling 추가
        if self.downsampling:
            skip_x = self.skip_conv(skip_x)

        x = self.first_conv(x)
        x = self.second_conv(x)

        x = x + skip_x
        x = self.relu(x)
        return x

class ResNet_mid(nn.Module):
    def __init__(self):
        super().__init__()
        # 입력 채널, 출력 채널, 블럭 갯수가 동일하게 들어감
        self.layer1 = self.make_layer(64, 64, 2)
        # 크기 조절 추가
        self.layer2 = self.make_layer(64, 128, 2, True)
        self.layer3 = self.make_layer(128, 256, 2, True)
        self.layer4 = self.make_layer(256, 512, 2, True)

    # 크기 조절 추가
    def make_layer(self, in_channel, out_channel, num_block, downsampling=False):
        layer = [resnet_block(in_channel, out_channel, downsampling)]
        for _ in range(num_block-1):
            layer.append(resnet_block(out_channel, out_channel))
        # 리스트 내부 layer 사용을 위해 언패킹
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.front = ResNet_front()
        self.middle = ResNet_mid()
        self.back = ResNet_back()

    def forward(self, x):
        x = self.front(x)
        x = self.middle(x)
        x = self.back(x)

        return x