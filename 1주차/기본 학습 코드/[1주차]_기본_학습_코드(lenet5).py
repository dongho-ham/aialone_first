"""
# 1주차에서 다루는 기본 학습 코드 

- 목표 : 학습 코드의 매우 기본적인 틀을 잡고 내것으로 만들기 (평가는 다음 시간에) 
- 주의사항 : 실시간 코딩으로 구체적인 구현체(함수, 변수명 등)의 차이가 있을 수 있음 


- 사용 데이터 : MNIST 
- 사용 모델 : LeNet5
"""

# 필요한 패키지 업로드 
import torch 
import torch.nn as nn 
from torch.optim import Adam 
from torchvision import datasets 
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import Resize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 하이퍼파라메터 세팅 
batch_size = 100 
num_classes = 10 
hidden_size = 500
lr = 0.001
epochs = 5 

image_size = 32

# MNIST 데이터 관련 (dataset, dataloader)
trans = Compose([
    Resize(image_size), 
    ToTensor(),
])

train_dataset = datasets.MNIST(root='./mnist', transform=trans, train=True, download=True)
test_dataset = datasets.MNIST(root='./mnist', transform=trans, train=False, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# shape 측정 
nn.Conv2d(3, 6, 5, 1)(torch.randn((3, 32, 32))).shape

# 모델 class 짜기 
class myLeNet5(nn.Module): 
    def __init__(self, num_classes): 
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.bn1(x) 
        x = self.relu(x) 
        x = self.pool(x) 
        x = self.conv2(x)
        x = self.bn2(x) 
        x = self.relu(x) 
        x = self.pool(x) 

        x = x.reshape(batch_size, -1)

        x = self.fc1(x)
        x = self.relu(x) 
        x = self.fc2(x)
        x = self.relu(x) 
        x = self.fc3(x)
        return x

# Loss와 Optimizer Class 짜기 
pass

# 모델, Loss, Optimizer 객체 만들기 
model = myLeNet5(num_classes).to(device)
criteria = nn.CrossEntropyLoss() 
optim = Adam(model.parameters(), lr=lr)

model.parameters

optim

# 학습 시각화 
losses = []

# 실제 학습 loop 돌리기 
for epoch in range(epochs) : 
    for idx, (image, label) in enumerate(train_loader) : 
        image = image.to(device)
        label = label.to(device) 

        output = model(image)
        loss = criteria(output, label)
        optim.zero_grad()
        loss.backward() 
        optim.step() 
        
        if idx % 100 == 0: 
            print(f'{epoch}/{epochs} , {idx} step | Loss : {loss.item():.4f}')
            losses.append(loss.item())

import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()

