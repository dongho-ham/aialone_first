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
import argparse

# parser 만들기
def parse_args():
    # 객체 생성
    # 모델 세팅 값을 인지해서 받음
    parser = argparse.ArgumentParser()
    # default는 hp를 설정하지 않아도 100을 넣어줌. 따로 설정하면 override됨.
    # type은 변수 값의 형태를 정해줌. int를 넣지 않으면 오류를 출력(안전장치)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--device', default=torch.device('cuda'if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--model', type=str, default='lenet', choices=['mlp','lenet','convs','linear','incep'])

    return parser.parse_args()

def main():
    # parser에 선언한 hp를 사용하기 위해 parse_args()함수 호출
    args = parse_args()

    train_dataset = CIFAR10(root='./cifar', train=True, download=True)
    
    std = train_dataset.data.std(axis=(0, 1, 2)) / 255.0
    mean = train_dataset.data.mean(axis=(0, 1, 2)) / 255.0

    trans = Compose([
        ToTensor(), Resize(args.image_size), Normalize(mean, std)
    ])

    train_dataset = CIFAR10(root='./cifar', train=True, transform=trans, download=True)
    test_dataset = CIFAR10(root='./cifar', train=False, transform=trans, download=True)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

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

    label = labels[label]
    # cv2.imshow(img, label)

    # 모델 class 
    class MyMLP(nn.Module): 
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

    class MyLeNet_incep(nn.Module):
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

    # model의 예측값과 실제 값을 저장하고 있는 dataloader를 함수 입력으로 사용
    def eval(model, dataloader):
        correct = 0
        total = 0
        for image, target in dataloader:
            image.to(args.device)
            target.to(args.device)

            output = model(image)

            _, pred = torch.max(output, 1)

            correct += (pred == target).sum().item()
            total += image.shape[0]
        return correct / total

    def eval_class(model, dataloader):
        correct = torch.zeros(args.num_classes)
        for idx, (image, target) in enumerate(dataloader):
            image.to(args.device)
            target.to(args.device)

            output = model(output)

            _, pred = torch.max(output, 1)

            correct[idx] += ((pred==idx) & (target==idx)).sum().item()
            target[idx] += (target==idx).sum().item()
        return correct, target

    # choices에 저장한 args를 통해 모델을 선택합니다.
    if args.model_type == 'lenet':
        model = MyLeNet(args.num_classes)
    elif args.model_type == 'mlp':
        model = MyMLP(args.hidden_size, args.num_classes)
    elif args.model_type == 'convs':
        model = MyLeNet_Conv(args.num_classes)
    elif args.model_type == 'linear':
        model = MyLeNet_linear(args.num_classes)
    elif args.model_type == 'incep':
        model = MyLeNet_incep(args.num_classes)

    loss = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for idx, (image, target) in enumerate(train_dataloader):
            image.to(args.device)
            target.to(args.device)

            output = model(image)
            val_loss = loss(output, target)

            optim.zero_grad()
            val_loss.backward()
            optim.step()

            if idx % 100 == 0:
                print(val_loss.item())
                # 디버깅할 때 왜 동일한 함수를 호출해야 하는지?
                print('accuracy: ', eval(model, test_dataloader))

# 우리의 전체 파일을 main 함수에 넣어줌
# 실제 프로그램이 제일 먼저 실행하는 부분
if __name__ == '__main__':
    main()
