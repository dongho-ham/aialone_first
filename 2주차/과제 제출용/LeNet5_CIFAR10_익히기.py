import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import Normalize
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 하이퍼파라미터 세팅 
batch_size = 100 
num_classes = 10 
hidden_size = 500
lr = 0.001
epochs = 5 
# LeNet을 사용하기 위한 img size hyper parameter
image_size = 32

# 데이터셋 불러오기
train_dataset = CIFAR10(root='./cifar', train=True, download=True)
# Normalize의 mean, std를 구하기 위한 코드 
# axis = (r, g, b)
mean = train_dataset.data.mean(axis=(0, 1, 2)) / 255.0
std = train_dataset.data.std(axis=(0, 1, 2)) / 255.0

# 이미지에 따라 특정 R, G, B 값이 클 수 있음. 한 속성이 다른 속성보다 너무 크게 되면 모델 일반화의 장애물이 될 수 있어 정규화를 실시함.
# Compose로 여러 이미지 변환을 한 번에 수행할 수 있음
# Resize를 통해 이미지 사이즈를 32*32로 변경, ToTensor로 0 ~ 255 크기의 픽셀을 0 ~ 1로 변경
trans = Compose([
    ToTensor(), Resize(image_size), Normalize(mean, std)
])

train_dataset = CIFAR10(root='./cifar', train=True, download=True, transform=trans)
test_dataset = CIFAR10(root='./cifar', train=False, download=True, transform=trans)
# 데이터 로더
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

# 원본 이미지 복구하기
# 데이터셋 이미지 확인하기 (tensor -> numpy)
def reverse_trans(x):
    x = (x * std) + mean
    return x.clamp(0, 1) * 255
# 원본 이미지를 반환하는 함수
def get_numpy_image(data): 
    img = reverse_trans(data.permute(1, 2, 0)).type(torch.uint8).numpy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 이미지 복구
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
idx = 1000

image, target = train_dataset.__getitem__(idx)

img = cv2.resize(
    get_numpy_image(image), (512,512)
)

label = labels[target]

# 기본 LeNet5
class MyLeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # LeNet5의 경우 32*32 사이즈의 이미지 한 개를 입력 받고, 6개 channel의 5*5 filter로 conv 연산을 수행
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)

        # 16장의 5 x 5 특성맵을 120개의 5 x 5 x 16 사이즈의 필터와 컨볼루션 해준다. 결과적으로 120개의 1 x 1 특성맵이 산출된다.
        self.fc1 = nn.Linear(16*5*5, 120)
        # 84개의 유닛을 가진 피드포워드 신경망이다.
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    # forward 함수가 내부적으로 어떻게 작동하고 연산되는지?
    def forward(self, x):
        # b, c, w, h = x.shape
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # x의 형상이 (batch_size, num_channels, height, width)인 경우 (batch_size, num_channels*height*width)
        # 2차원으로 변환해야 FC layer에 전달 가능
        x = x.reshape(batch_size, -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# LeNet5 sequential
class MyLeNet5_seq(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_seq1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            
        )
        # LeNet5의 경우 32*32 사이즈의 이미지 한 개를 입력 받고, 6개 channel의 5*5 filter로 conv 연산을 수행
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_seq2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_seq = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, num_classes)
        )

    # forward 함수가 내부적으로 어떻게 작동하고 연산되는지?
    def forward(self, x):
        # b, c, w, h = x.shape
        batch_size = x.shape[0]
        x = self.conv_seq1(x)
        x = self.conv_seq2(x)
        
        # x의 형상이 (batch_size, num_channels, height, width)인 경우 (batch_size, num_channels*height*width)
        # 2차원으로 변환해야 FC layer에 전달 가능
        x = x.reshape(batch_size, -1)

        x = self.fc_seq(x)
        return x

# LeNet5 Linear
class MyLeNet5_linear(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # LeNet5의 경우 32*32 사이즈의 이미지 한 개를 입력 받고, 6개 channel의 5*5 filter로 conv 연산을 수행
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=6)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)

        self.tmp_fc1 = nn.Linear(6*14*14, 2048)
        self.tmp_fc2 = nn.Linear(2048, 6*14*14)

        # 16장의 5 x 5 특성맵을 120개의 5 x 5 x 16 사이즈의 필터와 컨볼루션 해준다. 결과적으로 120개의 1 x 1 특성맵이 산출된다.
        self.fc1 = nn.Linear(16*5*5, 120)
        # 84개의 유닛을 가진 피드포워드 신경망이다.
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    # forward 함수가 내부적으로 어떻게 작동하고 연산되는지?
    def forward(self, x):
        # b, c, w, h = x.shape
        b = x.shape[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        _, tmp_c, tmp_w, tmp_h = x.shape
        x = x.reshape(-1, 6*14*14)
        x = self.tmp_fc1(x)
        x = self.tmp_fc2(x)
        x = x.reshape(b, tmp_c, tmp_w, tmp_h)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # x의 형상이 (batch_size, num_channels, height, width)인 경우 (batch_size, num_channels*height*width)
        # 2차원으로 변환해야 FC layer에 전달 가능
        x = x.reshape(b, -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# LeNet5 conv
class MyLeNet5_conv(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.ModuleList(
            # 사이즈를 줄인 후 이후 conv layer에 전달
            [nn.Conv2d(3, 6, 5, 1, 1)]+
            [nn.Conv2d(6, 6, 3, 1, 1) for _ in range(3)]
        )
        self.bn1 = nn.BatchNorm2d(num_features=6)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.ModuleList(
            [nn.Conv2d(6, 16, 5, 1)] + 
            [nn.Conv2d(6, 16, 3, 1) for _ in range(2)]
        )
        self.bn2 = nn.BatchNorm2d(num_features=16)

        self.fc_seq = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        # b, c, w, h = x.shape
        batch_size = x.shape[0]
        for module in self.conv1:
            x = module(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        for module in self.conv2:
            x = module(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # x의 형상이 (batch_size, num_channels, height, width)인 경우 (batch_size, num_channels*height*width)
        # 2차원으로 변환해야 FC layer에 전달 가능
        x = x.reshape(batch_size, -1)

        x = self.fc_seq(x)
        return x

# LeNet incep
class MyLeNet5_incep(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv_incep1 = nn.Conv2d(3, 6, 5, 1, 2)
        self.conv_incep2 = nn.Conv2d(3, 6, 3, 1, 1)
        self.conv_incep3 = nn.Conv2d(3, 6, 1, 1, 0)

        self.conv_seq1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=6),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_seq2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.fc_seq = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x_1 = self.conv_incep1(x)
        x_2 = self.conv_incep2(x)
        x_3 = self.conv_incep3(x)
        x_cat = torch.cat((x_1, x_2, x_3), dim=1)
        
        # b, c, w, h = x.shape
        batch_size = x.shape[0]
        x = self.conv_seq1(x_cat)
        x = self.conv_seq2(x)

        x = x.reshape(batch_size, -1)
        x = self.fc_seq(x)
        return x

def eval(model, dataloader):
    total = 0
    correct = 0
    for image, target in dataloader:
        image.to(device)
        target.to(device)

        output = model(image)

        _, pred = torch.max(output, 1)

        correct += (pred == target).sum().item()
        total += image[0]
    return correct / total

def eval_class(model, dataloader):
    total = torch.zeros(num_classes)
    correct = torch.zeros(num_classes)

    for idx, (image, target) in enumerate(dataloader):
        image.to(device)
        target.to(device)

        output = model(image)

        _, pred = torch.max(output, 1)

        correct += ((pred == idx) & (output == idx)).sum().item()
        total += (target == idx).sum().item()
    return correct, total

model = MyLeNet5(num_classes)
loss = nn.CrossEntropyLoss()
optim = Adam(model.parameters(), lr=lr)

step = 0
for epoch in range(epochs):
    for idx, (image, target) in enumerate(train_dataloader):
        if step==522:
            print(step)
        image.to(device)
        target.to(device)

        output = model(image)
        val_loss = loss(output, target)

        optim.zero_grad()
        val_loss.backward()
        optim.step()

        if idx%100==0:
            print(val_loss.item())
            print('accuracy:', eval(model, test_dataloader))
        step += 1




























