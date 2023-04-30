# 자주 사용하지 않는 기타 코드를 저장합니다. ex) dataset 변환 ...
import cv2
import torch
import os

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

# result 폴더와 하위 폴더의 경로를 생성하는 함수
def get_save_folder_path(args):
     # os.path.exists() method in Python is used to check whether the specified path exists or not.
     # args.save_folder의 default 값인 result 파일이 없다면 하위 코드 실행 
    if not os.path.exists(args.save_folder):
        # result folder 생성
        os.makedirs(args.save_folder)
        # 경로는 save_folder에 1 폴더 추가
        path = os.path.join(args.save_folder, '1')
        return path
    
    # 만약 폴더를 생성하고 실행을 멈췄을 때 최신 폴더 다음으로 생성하고 싶을 수 있음. ex)8 폴더 다음 9 폴더 생성...
    # os.listdir을 통해 args.save_folder 경로에 있는 파일/폴더 리스트를 가져옴
    current_max_value = max([int(f) for f in os.listdir(args.save_folder)])
    new_folder_name = str(current_max_value+1)
    # path에 result/1 or result2... 등의 경로가 생성됨.
    path = os.path.join(args.save_folder, new_folder_name)
    return path

# 가독성 더 좋게
def get_save_folder_path(args):
     # 폴더가 존재하지 않을 때 폴더를 생성하는 코드
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
        new_folder_name = '1'
    else:
        current_max_value = max([int(f) for f in os.listdir(args.save_folder)])
        new_folder_name = str(current_max_value+1)

    path = os.path.join(args.save_folder, new_folder_name)
    return path