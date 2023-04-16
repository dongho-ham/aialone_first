# 자주 사용하지 않는 기타 코드를 저장합니다. ex) dataset 변환 ...
import cv2
import torch

cifar_mean = [0.49139968, 0.48215827, 0.44653124]
cifar_std = [0.24703233, 0.24348505, 0.26158768]

def get_image(dataset, std, mean):

    def reverse_trans(x):
            x = (x * std) + mean
            return x.clamp(0, 1) * 255 

    def get_numpy_image(data): 
        img = reverse_trans(data.permute(1, 2, 0)).type(torch.uint8).numpy()
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    idx = 1000 

    img, label = dataset.__getitem__(idx)
    img = cv2.resize(
        get_numpy_image(img), 
        (512, 512)
    )
    label = labels[label]
    # cv2.imshow(label, img)

