"""
# 1주차에서 다루는 기본 학습 코드 

- 목표 : 학습 코드의 매우 기본적인 틀을 잡고 내것으로 만들기 (평가는 다음 시간에) 
- 주의사항 : 실시간 코딩으로 구체적인 구현체(함수, 변수명 등)의 차이가 있을 수 있음 


- 사용 데이터 : MNIST 
- 사용 모델 : MLP (명확한 논문이 있는게 아니니 아래의 구조로 사용) 
    - Hidden Layer : 3개 
    - Layer 당 Perceptron : 500개
"""

# 필요한 패키지 업로드 
import torch 
import torch.nn as nn 
from torch.optim import Adam 
from torchvision import datasets 
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 하이퍼파라메터 세팅 
batch_size = 100 
num_classes = 10 
hidden_size = 500
lr = 0.001
epochs = 5

# MNIST 데이터 관련 (dataset, dataloader)
train_dataset = datasets.MNIST(root='./mnist', transform=ToTensor(), train=True, download=True)
test_dataset = datasets.MNIST(root='./mnist', transform=ToTensor(), train=False, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# 모델 class 짜기 
class myNetwork(nn.Module): 
    def __init__(self, hidden_size, num_classes): 
        super().__init__()
        self.hidden_layer1 = nn.Linear(28*28, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.reshape(-1, 28*28)
        x = self.hidden_layer1(x) 
        x = self.hidden_layer2(x) 
        x = self.hidden_layer3(x) 
        return x

# Loss와 Optimizer Class 짜기 
pass

# 모델, Loss, Optimizer 객체 만들기 
model = myNetwork(hidden_size, num_classes).to(device)
criteria = nn.CrossEntropyLoss() 
optim = Adam(model.parameters(), lr=lr)

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

