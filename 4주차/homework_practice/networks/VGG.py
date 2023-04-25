import torch.nn as nn

class VGG_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = 1 if kernel_size == 3 else 0
        self.convs = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                               kernel_size=kernel_size, stride=1, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.convs(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class VGG_block(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, last_1conv=False):
        super().__init__()
        self.first_conv = VGG_conv(in_channels, out_channels)
        self.mid_conv = nn.ModuleList([
            VGG_conv(out_channels, out_channels) for _ in range(num_convs-2)
        ])
        kernel_size=1 if last_1conv else 3
        self.last_convs = VGG_conv(out_channels, out_channels, kernel_size=kernel_size)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.first_conv(x)
        for module in self.mid_conv:
            x = module(x)
        x = self.last_convs(x)
        return x

class VGG_classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.FC = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        x = self.FC(x)
        return x

class VGG_A(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.block1 = VGG_block(3, 64, 1)
        self.block2 = VGG_block(64, 128, 1)
        self.block3 = VGG_block(128, 256, 3)
        self.block4 = VGG_block(256, 512, 3)
        self.block5 = VGG_block(512, 512, 3)
        self.fc = VGG_classifier(num_classes)

    def forward(self, x):
        b, c, w, h = x.shape
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.reshape(b, -1)
        x = self.fc(x)
        return x

class VGG_B(VGG_A):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.block1 = VGG_block(3, 64, 3)
        self.block2 = VGG_block(64, 128, 3)

class VGG_C(VGG_B):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.block3 = VGG_block(128, 256, 3, True)
        self.block4 = VGG_block(256, 512, 3, True)
        self.block5 = VGG_block(512, 512, 3, True)

class VGG_D(VGG_C):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.block3 = VGG_block(128, 256, 4)
        self.block4 = VGG_block(256, 512, 4)
        self.block5 = VGG_block(512, 512, 4)

class VGG_E(VGG_D):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.block3 = VGG_block(128, 256, 5)
        self.block4 = VGG_block(256, 512, 5)
        self.block5 = VGG_block(512, 512, 5)