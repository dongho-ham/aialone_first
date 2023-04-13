# 패키지 불러오기 
import torch 
import torch.nn as nn 
from torch.optim import Adam
from torchvision.datasets import CIFAR10
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import Resize

import cv2

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
train_dataset = CIFAR10(root='./cifar', train=True, download=True)
mean = train_dataset.data.mean(axis=(0, 1, 2)) / 255.0
std = train_dataset.data.std(axis=(0, 1, 2)) / 255.0

transform = Compose([
    Resize((img_size, img_size)), 
    ToTensor(),
    Normalize(mean, std)
])

train_dataset = CIFAR10(root='./cifar', train=True, transform=transform, download=True)
test_dataset = CIFAR10(root='./cifar', train=False, transform=transform, download=True)

# dataloader 
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# dataset 이미지 확인하기 (tensor -> numpy) 
def reverse_trans(x):
    x = (x * std) + mean
    return x.clamp(0, 1) * 255

def get_numpy_image(data): 
    img = reverse_trans(data.permute(1, 2, 0)).type(torch.uint8).numpy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
idx = 1000

img, label = train_dataset.__getitem__(idx)
img = cv2.resize(
    get_numpy_image(img), 
    (512, 512)
)
label = labels[label]
# cv2.imshow(label, img)

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
        b, _, _, _ = x.shape
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

class myLeNet_seq(nn.Module):
    def __init__(self, num_classes): 
        super().__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5) ,
            nn.BatchNorm2d(num_features=6),
            nn.ReLU() ,

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) ,
            nn.BatchNorm2d(num_features=16),
            nn.ReLU() ,

            nn.MaxPool2d(kernel_size=2),
        )
        self.seq2 = nn.Sequential(
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Linear(in_features=120, out_features=84),
            nn.Linear(in_features=84, out_features=num_classes),
        ) 
        

    def forward(self, x): 
        b, c, h, w = x.shape
        x = self.seq1(x)
        x = x.reshape(b, -1)
        x = self.seq2(x)
        return x 

class myLeNet_linear(nn.Module):
    def __init__(self, num_classes): 
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5) 
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.act1 = nn.ReLU() 

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.fc_1 = nn.Linear(6*14*14, 2048)
        self.fc_2 = nn.Linear(2048, 6*14*14)

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

        _, tmp_c, tmp_w, tmp_h = x.shape
        x = x.reshape(b, -1) 
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = x.reshape(b, tmp_c, tmp_w, tmp_h) 

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x) 
        x = self.pool2(x)

        x = x.reshape(b, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x 

class myLeNet_convs(nn.Module):
    def __init__(self, num_classes): 
        super().__init__()
        self.tmp_conv1 = nn.ModuleList(
            [nn.Conv2d(3, 6, 3, 1, 1)] + [
            nn.Conv2d(6, 6, 3, 1, 1) for _ in range(3-1)
        ])
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=5) 
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
        for module in self.tmp_conv1: 
            x = module(x)
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

class myLeNet_incep(nn.Module):
    def __init__(self, num_classes): 
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 6, 5, 1, 2)
        self.conv1_2 = nn.Conv2d(3, 6, 3, 1, 1)
        self.conv1_3 = nn.Conv2d(3, 6, 1, 1, 0)

        self.conv1 = nn.Conv2d(in_channels=18, out_channels=6, kernel_size=5) 
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
        x_1 = self.conv1_1(x)
        x_2 = self.conv1_2(x)
        x_3 = self.conv1_3(x)

        x_cat = torch.cat((x_1, x_2, x_3), dim=1)
        x = self.conv1(x_cat)
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
model = myLeNet_incep(num_classes).to(device)
loss = nn.CrossEntropyLoss() 
optim = Adam(model.parameters(), lr=lr)

def eval(model, loader):
    total = 0 
    correct = 0 
    for idx, (image, target) in enumerate(loader):
        image = image.to(device)
        target = target.to(device)

        out = model(image)
        _, pred = torch.max(out, 1)

        correct += (pred == target).sum().item()
        total += image.shape[0]
    return correct / total 

def eval_class(model, loader):
    total = torch.zeros(num_classes) 
    correct = torch.zeros(num_classes) 
    for idx, (image, target) in enumerate(loader):
        image = image.to(device)
        target = target.to(device)

        out = model(image)
        _, pred = torch.max(out, 1)

        for i in range(num_classes): 
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
            print('accuracy : ', eval(model, test_loader))
