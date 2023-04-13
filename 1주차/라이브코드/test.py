# 패키지 불러오기
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam

# 하이퍼파라미터
batch_size = 100
hidden_size = 500
num_classes = 10
lr = 0.001
epochs = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터 불러오기
# dataset

# root='./mnist': 데이터셋을 다운로드할 경로를 지정. 현재 작업 디렉토리 안에 mnist 폴더를 지정
# transform=ToTensor: 다운로드한 이미지를 모델의 input으로 사용하기 위해 Tensor로 변환
# download=True: 데이터셋이 존재하지 않을 때 다운로드 진행. False로 지정했을 때 데이터셋이 없으면 오류 발생
train_dataset = MNIST(root='./mnist', train=True, transform=ToTensor(), download=True)
test_dataset = MNIST(root='./mnist', train=False, transform=ToTensor(), download=True)

# dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 전처리

# 모델 설계
class myMLP(nn.Module):
    # 초기화 과정
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        # torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None), support TensorFloat32
        self.fc1 = nn.Linear(28*28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # flattening을 해주는 과정
        # batch_size, channel, width, height
        b, c, w, h = x.shape # 100, 1, 28, 28
        x = x.reshape(b, 28*28) # 100, 28x28
        # x = x.reshape(-1, 28*28)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x

# 학습 세팅
# 모델, loss, optimizer
model = myMLP(hidden_size, num_classes).to(device)
loss = nn.CrossEntropyLoss()
# parameters에 ()를 써야 weight와 bias가 전달됨.
optim = Adam(model.parameters(), lr = lr)

# 학습 loop
# MNIST 검색 후 __get_item__을 확인하면 data_loader가 image와 target을 가져오게 됨.
for epoch in range(epochs):
    for idx, (image, target) in enumerate(train_loader):
        # 일반적으로 학습할 때는 img와 target tensor를 GPU로 이동시켜 학습 속도를 높임. cpu면 cpu, gpu면 gpu 동일하게 사용해야 함.
        image = image.to(device)
        target = target.to(device)

        out = model(image)
        # loss(모델 출력, 정답), loss_val에는 예측 값과 실제 값의 차이인 손실 값이 존재함.
        loss_val = loss(out, target)
        # 시험 과정에서는 업데이트한 w 삭제
        optim.zero_grad()
        # 손실 값을 가중치로 미분(편미분)하면 가중치가 손실 값을 어떻게 변화시키는지 알 수 있음. 모델의 학습 중에 필수적으로 호출되는 함수이며,
        # 모델의 매개변수에 대한 gradient를 계산함. backward를 실행하면 역전파가 시작됨.
        loss_val.backward()
        # backward() 메서드를 호출하여 계산된 gradient를 바탕으로 weight을 업데이트, backward() 메서드와 함께 사용해 가중치를 업데이트.
        optim.step()
        
        # 100번에 한 번씩 loss 출력. logging 단계
        if idx % 100 == 0:
            print(loss_val.item())
            # tensor([1.42423], grad_fn=~~~) -> tensor 형태로 저장된 손실 값이 실수형으로 변환되서 출력됨.