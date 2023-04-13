# LeNet5 기본 코드 및 평가 코드 추가
# 패키지 불러오기 
import torch 
import torch.nn as nn 
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import Resize

# 하이퍼파라메터
batch_size = 100 
hidden_size = 500 
num_classes = 10
lr = 0.001
epochs = 3 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_size = 32 

# 데이터 불러오기 
# dataset 
# 이미지 크기 변경 & tensor로 변경 
transform = Compose([
    Resize((img_size, img_size)), 
    ToTensor(),
])

train_dataset = MNIST(root='./mnist', train=True, transform=transform, download=True)
test_dataset = MNIST(root='./mnist', train=False, transform=transform, download=True)

# dataloader 
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 전처리 

# 모델 class 
class myMLP(nn.Module): 
    def __init__(self, hidden_size, num_classes): 
        super().__init__()
        self.fc1 = nn.Linear(28*28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        b, c, w, h = x.shape  # 100, 1, 28, 28  
        x = x.reshape(-1, 28*28) # 100, 28x28 
        # x = x.reshape(b, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

class myLeNet(nn.Module):
    def __init__(self, num_classes): 
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5) 
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
    
# 모델, loss, optimizer 
# model = myMLP(hidden_size, num_classes).to(device)
model = myLeNet(num_classes).to(device)
loss = nn.CrossEntropyLoss() 
optim = Adam(model.parameters(), lr=lr)

# 전체 예측값 중 실제로 맞은 개수
# model의 예측값과 실제 값을 저장하고 있는 dataloader를 함수 입력으로 사용
def eval(model, loader):
    total = 0 
    correct = 0 
    for idx, (image, target) in enumerate(loader):
        image = image.to(device)
        target = target.to(device)

        out = model(image)
        # max(input, dim=1)을 사용한 이유: class 별로 예측한 확률 값 중 가장 큰 값을 반환하기 위해
        # pred.shape =  torch.Size([100])
        # _에 행별 가장 큰 확률 값이 저장되고, pred에 확률 값은 index가 저장됨.
        _, pred = torch.max(out, 1)
        # pred와 target이 같으면 1(True)이 반환됨.
        correct += (pred == target).sum().item()
        # total에는 반복할 때마다 100씩 더해짐
        # image.shape = torch.Size([100, 1, 32, 32])
        total += image.shape[0]
    return correct / total

# class 별 정답
# 길이가 9인 리스트를 만들어 리스트 순서대로 class 별 몇 개의 정답이 있는지 맞출 것임.
def eval_class(model, loader):
    total = torch.zeros(num_classes) 
    correct = torch.zeros(num_classes) 
    for i, (image, target) in enumerate(loader):
        image = image.to(device)
        target = target.to(device)

        out = model(image)
        _, pred = torch.max(out, 1)

        for i in range(num_classes):
            # target과 pred가 같을 때만 correct의 i번째 index에 저장
            # batch 100개 중 0~9까지 순환해서 같으면 True(1), 다르면 False(0)을 반환하고, 합하면 전체 개수가 나옴. 
            correct[i] += ((target == i) & (pred == i)).sum().item()
            total[i] += (target == i).sum().item()
    return correct, total

# 학습 loop 
for epoch in range(epochs): 
    for idx, (image, target) in enumerate(train_loader):
        image = image.to(device)
        target = target.to(device)
        
        out = model(image)
        loss_value = loss(out, target)
        optim.zero_grad() 
        loss_value.backward()
        optim.step()

        if idx % 100 == 0 : 
            print(loss_value.item())
            print('accuracy : ', eval_class(model, test_loader))
