import torch.nn as nn
import torch

# resnet 입력층
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
    
# resnet 출력층
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

# resnet 구현을 위한 개별 block
class resnet_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.first_conv = nn.Sequential(
            # 두 번째 채널로 넘어갈 때 크기가 줄어들면 안되기 때문에 3, 1, 3으로 설정.
            nn.Conv2d(in_channel, out_channel, 3, 1, 3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.second_conv = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        # 논문에서 skip connection을 적용하고 relu를 통과시킴. 필요한 relu 선언
        self.relu = nn.ReLU()

    def forward(self, x):
        skip_x = torch.clone(x)

        x = self.first_conv(x)
        x = self.second_conv(x)
        # skip connection
        x = x + skip_x
        x = self.relu(x)
        return x

# resnet 중간층
# config 18을 기준으로 구현
class ResNet_mid(nn.Module):
    def __init__(self):
        super().__init__()
        # 입력 채널, 출력 채널, 블럭 갯수가 동일하게 들어감
        self.layer1 = self.make_layer(64, 64, 2)
        self.layer2 = self.make_layer(64, 128, 2)
        self.layer3 = self.make_layer(128, 256, 2)
        self.layer4 = self.make_layer(256, 512, 2)

    # 개별 layer를 만들기 위한 함수 정의
    def make_layer(self, in_channel, out_channel, num_block):
        # layer = []
        # layer.append(ResNet_block(64, 64))
        # layer.append(ResNet_block(64, 64))
        # layer.append(ResNet_block(64, 128))
        # layer.append(ResNet_block(128, 128))
        # layer.append(ResNet_block(128, 256)) ...
        layer = [layer.append(resnet_block(in_channel, out_channel))]
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
    

