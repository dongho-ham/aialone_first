# 패키지 불러오기
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision.transforms import Normalize
from torch.optim import Adam
from torch.utils.data import DataLoader
import cv2

# 하이퍼 파라미터
batch_size = 100 
hidden_size = 500 
num_classes = 10
lr = 0.001
epochs = 3 
device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

# LeNet을 사용하기 위한 img size hyper parameter
img_size = 32

# 전처리
# 이미지 크기 변경 & tensor로 변경
# axis = (r, g, b)
train_dataset = CIFAR10(root='./cifar', train=True, download=True)
# 이미지에 따라 특정 R, G, B 값이 클 수 있음. 한 속성이 다른 속성보다 너무 크게 되면 모델 일반화의 장애물이 될 수 있어 정규화를 실시함. 
std = train_dataset.data.std(axis=(0, 1, 2)) / 255.0
mean = train_dataset.data.mean(axis=(0, 1, 2)) / 255.0

trans = Compose([
    ToTensor(), Resize(img_size), Normalize(mean, std)
])

train_dataset = CIFAR10(root='./cifar', train=True, transform=trans, download=True)
test_dataset = CIFAR10(root='./cifar', train=False, transform=trans, download=True)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

def reverse_trans(x):
    x = (x*std) + mean
    return x.clamp(0, 1)*255

def image_to_numpy(data):
    img = reverse_trans(data.permute(1, 2, 0).type(torch.uint8)).numpy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
idx = 1000

img, label = train_dataset.__getitem__(idx)

img = cv2.resize(
    image_to_numpy(img),
    (512, 512)
)

labe = labels[label]

class MyLeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=6)

        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)

        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(x)
        x = self.relu(x)

        b, c, w, h = x.shape
        x = x.reshape(b, -1)
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

class MyLeNet_Conv(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.tmp_conv1 = nn.ModuleList(
            [nn.Conv2d(3, 6, 3, 1, 1)] +
            [nn.Conv2d(6, 6, 3, 1, 1)]
        )

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=6)

        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)

        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(x)
        x = self.relu(x)

        b, c, w, h = x.shape
        x = x.reshape(b, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x

# model의 예측값과 실제 값을 저장하고 있는 dataloader를 함수 입력으로 사용
def eval(model, dataloader):
    correct = 0
    total = 0
    for image, target in dataloader:
        image.to(device)
        target.to(device)

        output = model(image)

        _, pred = torch.max(output, 1)

        correct += (pred == target).sum().item()
        total += image.shape[0]
    return correct / total

def eval_class(model, dataloader):
    correct = torch.zeros(num_classes)
    for idx, (image, target) in enumerate(dataloader):
        image.to(device)
        target.to(device)

        output = model(output)

        _, pred = torch.max(output, 1)

        correct[idx] += ((pred==idx) & (target==idx)).sum().item()
        target[idx] += (target==idx).sum().item()
    return correct, target

model = MyLeNet(num_classes).to(device)
loss = nn.CrossEntropyLoss()
optim = Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    for idx, (image, target) in enumerate(train_dataloader):
        image.to(device)
        target.to(device)

        output = model(image)
        val_loss = loss(output, target)

        optim.zero_grad()
        val_loss.backward()
        optim.step()

        if idx % 100 == 0:
            print(val_loss.item())
            # 디버깅할 때 왜 동일한 함수를 호출해야 하는지?
            print('accuracy: ', eval(model, test_dataloader))

