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
mean = train_dataset.data.mean(axis=(0,1,2)) / 255.0
std = train_dataset.data.std(axis=(0,1,2)) / 255.0
# 이미지에 따라 특정 R, G, B 값이 클 수 있음. 한 속성이 다른 속성보다 너무 크게 되면 모델 일반화의 장애물이 될 수 있어 정규화를 실시함.
# Compose로 여러 이미지 변환을 한 번에 수행할 수 있음
# Resize를 통해 이미지 사이즈를 32*32로 변경, ToTensor로 0 ~ 255 크기의 픽셀을 0 ~ 1로 변경
trans = Compose([
    Resize(image_size),
    ToTensor(),
    Normalize(mean, std)
])

test_dataset = CIFAR10(root='./cifar', train=False, transform=trans, download=True)
train_dataset = CIFAR10(root='./cifar', train=True, transform=trans, download=True)
# 데이터 로더
train_dataloader = DataLoader(dataset = train_dataset, shuffle=True, batch_size = batch_size)
test_dataloader = DataLoader(dataset = test_dataset, shuffle=True, batch_size=batch_size)

# 데이터셋 이미지 확인하기 (tensor -> numpy)
# trans 과정을 거쳤기 때문에 이미지를 바로 확인하면 이상한 이미지가 나오게 됨. 원래 이미지를 보기 위해 픽셀 값을 원래대로 변환함.
# clamp 함수를 사용해서 0 ~ 1 사이의 범위로 x 값 변환
def reverse_trans(x):
    x = (x * std) + mean
    return x.clamp(0, 1) * 255

# 원본 이미지를 반환하는 함수
def get_numpy_image(data):
    # premute 함수를 통해 차원을 맞교환. OpenCV를 사용해서 이미지를 보기 위해서는 차원을 교환해야 함.
    # PIL을 tensor로 변경하면 shape은 c, h, w이지만 opencv의 shape은 h, w, c라 permute 함수롤 통해 차원 맞교환
    # tensor의 dtype은 기본적으로 float32, cv2의 dtype은 uint8
    # cv2.cvtColor의 데이터 타입은 numpy로 제한되어 있어 numpy 변환 필요 
    img = reverse_trans(data.permute(1, 2, 0)).type(torch.uint8).numpy()
    # cv2.cvtColor는 cv2의 색상 변환 함수
    # OpenCV가 이미지를 BGR로 읽기 때문에 원본 이미지와 다르게 색감이 나타날 수 있음.
    # 따라서 cv2.COLOR_BGR2RGB로 RGB로 변환해야 정상적인 색감을 확인할 수 있음.
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 이미지 복구
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
idx = 1000

# train_dataset의 idx번째 데이터를 transform을 적용해서 가져옴.
img, label = train_dataset.__getitem__(idx)
img = cv2.resize(
    # 이미지를 numpy 형태로 변환
    get_numpy_image(img),
    # 512*512로 사이즈 변환
    (512, 512)
)
# 이미지에서 가져온 label은 0 ~ 9사이 숫자로 labels의 인덱스에 접근해 이미지에 맞는 label이 지정됨.
label = labels[label]

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

# nn.Sequential 기능으로 forward 함수 깔끔하게 만들기
class MyLeNet5_seq(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.seq_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
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
    
# Yolo v1 (linear 함수 추가)
class MyLeNet_linear(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.seq_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
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
        # conv layer를 통과해 변형된 c, w, h를 저장
        _, tmp_c, tmp_w, tmp_h = x.shape

        x = x.reshape(b, 6*14*14)
        x = self.fc_mid1(x)
        x = self.fc_mid2(x)
        # FC layer를 통과하기 전, conv layer를 통과한 후 크기로 만들어줘야 하기 때문에 tmp 사용
        x = x.reshape(b, tmp_c, tmp_w, tmp_h)

        x = self.seq_conv2(x)

        b, c, w, h = x.shape
        x = x.reshape(b, -1)
        x = self.seq_fc(x)
        
        return x

# conv2 layer 추가
class MyLeNet_Conv(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.conv_seq1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=6, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=6),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        # RGB channel 크기로 in_channels =3, 크기 유지를 위해 kernel_size = 3 사용
        self.tmp_conv1 = nn.ModuleList(
            [nn.Conv2d(3, 6, 3, 1, 1)] +
            [nn.Conv2d(6, 6, 3, 1, 1) for _ in range(2)]
        )
        # conv_seq1에서 출력된 (6, 14, 14) image를 받을 수 있는 conv layer를 2개 추가
        # 마지막 layer는 conv_seq2의 입력으로 사용될 수 있게 16차원으로 변경
        self.tmp_conv2 = nn.ModuleList(
            [nn.Conv2d(6, 6, 3, 1, 1)] +
            [nn.Conv2d(6, 16, 3, 1, 1)]
        )

        self.conv_seq2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=16),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.fc_seq = nn.Sequential(
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Linear(in_features=120, out_features=84),
            nn.Linear(in_features=84, out_features=num_classes)
        )

    def forward(self, x):
        # module을 반복문을 이용해서 리스트에서 1개씩 추출
        for module in self.tmp_conv1:
            x = module(x)
        x = self.conv_seq1(x)

        for module in self.conv_seq2:
            x = module(x)
        x = self.conv_seq2(x)

        b, c, w, h = x.shape
        x = x.reshape(b, -1)
        x = self.fc_seq(x)
        
        return x

# conv 병합
class MyLeNet5_incep(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 3개의 다른 conv layer
        self.conv_incep1 = nn.Conv2d(3, 6, 5, 1, 2)
        self.conv_incep2 = nn.Conv2d(3, 6, 3, 1, 1)
        self.conv_incep3 = nn.Conv2d(3, 6, 1, 1, 0)

        # LeNet5의 경우 32*32 사이즈의 이미지 한 개를 입력 받고, 6개 channel의 5*5 filter로 conv 연산을 수행
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

        # 16장의 5 x 5 특성맵을 120개의 5 x 5 x 16 사이즈의 필터와 컨볼루션 해준다. 결과적으로 120개의 1 x 1 특성맵이 산출된다.
        self.fc_seq = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, num_classes)
        )

    # forward 함수가 내부적으로 어떻게 작동하고 연산되는지?
    def forward(self, x):
        x_1 = self.conv_incep1(x)
        x_2 = self.conv_incep2(x)
        x_3 = self.conv_incep3(x)
        x_cat = torch.cat((x_1, x_2, x_3), dim=1)
        
        # b, c, w, h = x.shape
        batch_size = x.shape[0]
        x = self.conv_seq1(x_cat)
        x = self.conv_seq2(x)
        
        # x의 형상이 (batch_size, num_channels, height, width)인 경우 (batch_size, num_channels*height*width)
        # 2차원으로 변환해야 FC layer에 전달 가능
        x = x.reshape(batch_size, -1)
        x = self.fc_seq(x)
        return x


# 모델 Loss, Optim 객체 만들기
model = MyLeNet5(num_classes).to(device)
Loss = nn.CrossEntropyLoss()
optim = Adam(model.parameters(), lr = lr)

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

# 평가코드 작성하기
def eval_class(model, loader):
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


# len(train_dataloader) = 500
# 523번째 loss를 확인하기 위해 step 변수 선언.
# train_dataloader의 길이가 500이므로 epoch가 1이 되었을 때 523번째 loss를 확인할 수 있음.
# if문을 만족할 때 중단점이 멈추게 되어 있음. 하지만 중단점이 멈췄다고 해서 val_loss를 확인하면 500번째 val_loss 값이 나옴.
# step over를 사용해 val_loss를 실행시킨 후 디버그 콘솔에서 확인 가능함.
step = 0
for epoch in range(epochs):
    for i, (image, target) in enumerate(train_dataloader):
        if step==522:
            # step에 중단점을 걸기위해 사용
            print(step)
        image.to(device)
        target.to(device)

        output = model(image)
        val_loss = Loss(output, target)

        optim.zero_grad()
        val_loss.backward()
        optim.step()

        if i%100==0:
            print(val_loss.item())
            print('accuracy:', eval(model, test_dataloader))
        step += 1