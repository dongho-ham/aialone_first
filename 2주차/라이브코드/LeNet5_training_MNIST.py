# 내가 작성한 baseline 코드
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import Resize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 하이퍼파라미터 세팅 
batch_size = 100 
num_classes = 10 
hidden_size = 500
lr = 0.001
epochs = 5 

image_size = 32

# Compose로 여러 이미지 변환을 한 번에 수행할 수 있음
# Resize를 통해 이미지 사이즈를 32*32로 변경, ToTensor로 0 ~ 255 크기의 픽셀을 0 ~ 1로 변경
trans = Compose([
    Resize(image_size),
    ToTensor()
])

# 데이터셋 불러오기
train_dataset = MNIST(root='./mnist', train=True, transform=trans, download=True)
test_dataset = MNIST(root='./mnist', train=False, transform=trans, download=True)

# 데이터 로더
train_dataloader = DataLoader(dataset = train_dataset, shuffle=True, batch_size = batch_size)
test_dataloader = DataLoader(dataset = test_dataset, shuffle=True, batch_size=batch_size)

# shape 측정
# 32*32 사이즈의 이미지 3개 생성 후 5*5 filter와 6 channel로 conv 연산
# nn.Conv2d(in_channel, out_channel, kernel_size, stride)
# torch.Size([6, 28, 28])
nn.Conv2d(3, 6, 5, 1)(torch.randn((3, 32, 32))).shape

class MyLeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # LeNet5의 경우 32*32 사이즈의 이미지 한 개를 입력 받고, 6개 channel의 5*5 filter로 conv 연산을 수행
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
         # batch norm 같은 경우 분산과 평균을 추적해서 왠만하면 따로 만드는게 좋음.
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
        # b, w, h = x.shape
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

class MyLeNet_seq(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.seq_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=6),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=16)
        )

        self.seq_fc = nn.Sequential(
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Linear(in_features=120, out_features=84),
            nn.Linear(in_features=84, out_features=num_classes)
        )

    def forward(self, x):
        x = self.seq_conv(x)
        b, c, w, h = x.shape
        x = x.reshape(b, -1)
        x = self.seq_fc(x)
        
        return x

class MyLeNet_linear(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.seq_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=6),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.fc_mid1 = nn.Linear(6*14*14, 2048)
        self.fc_mid2 = nn.Linear(2048, 6*14*14)

        self.seq_conv2 = nn.Sequential(
             nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
             nn.BatchNorm2d(num_features=16),
             nn.ReLU(),
             nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.seq_fc = nn.Sequential(
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Linear(in_features=120, out_features=84),
            nn.Linear(in_features=84, out_features=num_classes)
        )

    def forward(self, x):
        b, c, w, h = x.shape
        x = self.seq_conv1(x)

        _, tmp_c, tmp_w, tmp_h = x.shape
        x = x.reshape(b, 6*14*14)
        x = self.fc_mid1(x)
        x = self.fc_mid2(x)
        x = x.reshape(b, tmp_c, tmp_w, tmp_h)

        x = self.seq_conv2(x)

        b, c, w, h = x.shape
        x = x.reshape(b, -1)
        x = self.seq_fc(x)
        
        return x 



# 모델 Loss, Optim 객체 만들기
model = MyLeNet5(num_classes).to(device)
Loss = nn.CrossEntropyLoss()
optim = Adam(model.parameters(), lr = lr)

# 평가코드 작성하기
# 전체 데이터 중 맞춘 개수
# model의 예측값과 실제 값을 저장하고 있는 dataloader를 함수 입력으로 사용
def eval(model, loader):
    correct = 0
    total = 0

    for idx, (image, target) in enumerate(train_dataloader):
        image.to(device)
        target.to(device)

        output = model(image)
        # max(input, dim=1)을 사용한 이유: class 별로 예측한 확률 값 중 가장 큰 값을 반환하기 위해
        # pred.shape =  torch.Size([100])
        # _에 행별 가장 큰 확률 값이 저장되고, pred에 그에 따른 class가 저장됨.
        _, pred = torch.max(output, 1)
        # pred와 target이 같으면 1(True)이 반환됨.
        correct += (pred == target).sum().item()
        # total에는 반복할 때마다 100씩 더해짐
        total += image.shape[0]
    return correct / total

# 평가코드 작성하기
def eval_class(mode, loader):
    total = torch.zeros(num_classes)
    correct = torch.zeros(num_classes)
    for idx, (image, target) in enumerate(loader):
        image = image.to(device)
        target = target.to(device)

        out = model(image)
        _, pred = torch.max(out, 1)
        # 같으면 True(1), 틀리면 False(0)
        for i in range(num_classes):
            correct[i] += ((target == i) * (pred == i)).sum().item()
            total[i] += (target == 1).sum().item()

    return correct / total
    # return accuracy


for epoch in range(epochs):
    for idx, (image, target) in enumerate(train_dataloader):
        image = image.to(device)
        target = target.to(device)

        output = model(image)
        loss = Loss(output, target)
        
        # model(image)로 순전파를 했으면 역전파 하기 전에 가중치 초기화
        optim.zero_grad()
        loss.backward()
        optim.step()

        if idx % 100 == 0:
            print(loss.item())
            print('accuracy: ', eval_class(model, train_dataloader))
