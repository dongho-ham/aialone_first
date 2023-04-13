# LeNet5 기본 코드
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
