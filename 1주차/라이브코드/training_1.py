# 추가적으로 학습한 부분을 주석에 추가했습니다.
# 모듈 불러오기
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam

# 하이퍼 파라미터 설정하기
batch_size = 100
hidden_unit = 500
num_classes = 10
lr = 0.001
epochs = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')

# 데이터 불러오기
# 데이터셋 불러오기
# train data를 생성합니다. 데이터 셋은 MNIST에서 다운받고, 'mnist' 폴더에 저장됩니다. train=True를 사용해 학습용 데이터로 지정합니다.
# download = True를 사용해 데이터가 존재하지 않을 경우 다운로드 할 것을 알려줍니다.
# ToTensor를 사용해 이미지 데이터를 model input으로 사용할 수 있게 변환합니다.
train_data = MNIST(root='./mnist', train=True, transform = ToTensor, download=True)
# test data를 생성합니다. 데이터 셋은 MNIST에서 다운받고, 'mnist' 폴더에 저장됩니다. train=True를 사용해 학습용 데이터로 지정하지 않습니다.
test_data = MNIST(root='./mnist', train=False, transform = ToTensor, download=True)

# 데이터로더 사용하기
# torch.utils.data.DataLoader는 Dataset를 샘플에 쉽게 접근할 수 있는 순회 가능한 객체로 바꿉니다. 학습 시 loop를 통해 이미지와 target을 한 개씩 불러올 수 있습니다.
# shuffle=True를 통해 데이터를 무작위로 가져옵니다.
# batch_size=batch_size를 통해 하이퍼 파라미터로 설정한 batch_size 크기만큼의 데이터를 가져옵니다.
train_dataloader = DataLoader(dataset=train_data, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(dataset=test_data, shuffle=True, batch_size=batch_size)

# 모델 설계하기
class MyMLP(nn.Module):
    # MyMLP 모델을 초기화합니다. 객체를 생성할 때 자동으로 호출됩니다. model을 처음 생성할 때 hidden_size, num_classes를 입력해야 합니다.
    def __init__(self, hidden_size, num_classes):
        # nn.Module을 상속받기 위해 super()를 사용합니다. 이 클래스의 초기값은 nn.Module의 초기값으로 설정됩니다.
        super().__init__()
        # Multi Layer Perceptron 모델을 만들 때 사용해는 Linear 함수는 input dim과 output dim 값을 필수적으로 arg로 입력 받습니다.
        self.fc1 = nn.Linear(28*28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_classes)
    # forward 함수는 학습 데이터를 받으면 연산을 진행시키는 함수입니다. forward 함수는 객체를 데이터와 함께 호출하면 자동으로 실행되는 메서드입니다.
    def forward(self, x):
        # 28*28 사이즈의 데이터를 input으로 넣기 위해 flattening 작업을 실시합니다.
        b, c, w, h = x.shape
        x = x.reshape(b, 28*28)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
    
# 학습 세팅하기
# gpu를 사용할 것인지 cpu를 사용할 것인지 선택합니다
model = MyMLP(hidden_unit, num_classes).to(device)
# 손실함수를 정의합니다. 분류 문제이므로 Crossentropy 함수를 사용합니다.
loss = nn.CrossEntropyLoss()
# 최적의 parameter를 찾기 위해 optimizer로 Adam을 정의합니다. 기본적으로 model의 파라미터와 learing rate를 지정해줘야 합니다.
optim = Adam(model.parameters(), lr = lr)

# 학습하기
# epoch만큼 반복하기 위해 range(epochs)를 사용합니다.
for epoch in range(epochs):
    # 위에서 설명한대로 dataloader는 dataset를 iterable한 객체로 감쌉니다. 객체에서 하나 씩 꺼낼 경우 image와 target으로 꺼낼 수 있습니다.
    for idx, (image, target) in enumerate(train_dataloader):
        # 두 데이터 모두 동일한 device를 사용해야 합니다.
        image.to(device)
        target.to(device)
        # train data의 이미지를 모델에 넣어 output을 저장합니다.
        out = model(image)
        # 앞서 정의한 loss 함수에 예상 값과 정답을 비교해서 오차를 계산합니다.
        val_loss = loss(out, target)
        # pytorch에서는 gradients 값들을 추후에 backward할 때 계속 더해주기 때문에 backpropagation할 때 gradients를 zero로 만들어줘야 합니다.
        # 이상적인 학습을 위해서는 학습이 끝나고 gradient를 초기화하고 그렇지 않으면 gradient가 누적되어 값이 튈 수 있습니다.
        optim.zero_grad()
        # 모델의 매개변수에 대한 gradient를 계산하며 역전파하는 단계입니다.
        val_loss.backward()
        # backward를 사용해 계산된 gradient를 바탕으로 weight을 업데이트합니다.
        optim.step()

        # 100번에 한 번씩 loss를 출력합니다.
        if idx % 100 == 0:
            # tensor([1.42423], grad_fn=~~~) -> tensor 형태로 저장된 손실 값이 실수형으로 변환되서 출력됩니다.
            print('val loss = ', val_loss.item())