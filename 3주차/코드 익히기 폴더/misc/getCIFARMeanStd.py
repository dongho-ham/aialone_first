# CIFAR data의 mean과 std를 저장하는 파일입니다.

from torchvision.datasets import CIFAR10

train_dataset = CIFAR10(root='./cifar', train=True, download=True)
mean = train_dataset.data.mean(axis=(0, 1, 2)) / 255.0
std = train_dataset.data.std(axis=(0, 1, 2)) / 255.0