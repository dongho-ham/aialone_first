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

# 이번 과제로 만든 Custom LeNet5입니다.
# FC 1 층 통과 후 conv 연산 후 다음 FC layer들을 통과합니다.
# FC layer를 중간에 추가하는 것에 영감을 얻어 FC layer들 사이에 conv layer를 추가했습니다.
class MyLeNet5_custom(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # LeNet5의 경우 32*32 사이즈의 이미지 한 개를 입력 받고, 6개 channel의 5*5 filter로 conv 연산을 수행
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=6)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)

        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(num_features=8)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16*6*6, 16*6*6)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)

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
        # x.shape = 100, 16, 6, 6
        
        # x의 형상이 (batch_size, num_channels, height, width)인 경우 (batch_size, num_channels*height*width)
        # 2차원으로 변환해야 FC layer에 전달 가능
        x = x.reshape(batch_size, -1)

        x = self.fc1(x)

        x = x.reshape(batch_size, 16, 6, 6)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x= self.pool(x)

        x = x.reshape(batch_size, -1)

        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
model = MyLeNet5_custom(num_classes).to(device)
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