# 패키지 불러오기
from utils.parser import infer_parser_args
from utils.parser import load_trained_args
from utils.getModules import get_target_module, getTransform
from  PIL import Image
import torch.nn.functional as F
import os
import torch


def main():

    # 추론 전용 parser 불러오기
    args = infer_parser_args()
    # assert는 조건을 만족하지 않으면 조건 생성
    # 해당 경로에 폴더가 없으면 AssertionError가 발생하고 프로그램 중단
    assert os.path.exits(args.folder), "학습 폴더 생성"
    assert os.path.exits(args.iamge), "추론할 이미지 넣기"

    # 학습이 된 폴더를 기반으로 학습된 args 불러오기
    trained_args = load_trained_args()

    # 모델을 학습된 상황에 맞게 재설정
    # get_target_module에서는 학습에서 사용했던 args가 필요
    model = get_target_module(trained_args)

    # 모델 weight 업데이트
    model.load_state_dict(torch.load(os.path.join(args.folder, "best_model.ckpt")))

    # 데이터 전처리 코드
    transform = getTransform(trained_args)

    # 이미지 불러오기
    img = Image.open(args.image)
    img = transform(img)
    img = img.unsqueeze(0)

    # 모델 출력
    output = model(img)

    # 결과 후처리
    # socre로 나온 값을 확률 값으로 만들어줌. 이미지에 대한 10개 점수가 나옴
    prob = F.softmax(output, dim=1)
    index = torch.argmax(prob)
    value = torch.max(prob)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print(f'Image is {classes[index]}, and the confidence is {value*100:.2f}%')

if __name__ == '__main__':
    main()