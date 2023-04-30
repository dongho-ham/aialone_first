# import error 방지를 위한 패키지
import os
import sys
sys.path.append(os.getcwd())
import json

import torch
import torch.nn as nn 
from torch.optim import Adam

from utils.getModules import get_dataloader, get_target_module
from utils.parser import parser_args
from utils.evaluation import eval
# 함수 호출
from misc.tools import get_save_folder_path

def main():
    # f12 누르면 정의 파일로 이동
    args = parser_args()

    # 저장 폴더를 설정, args에 있는 result 폴더를 받음
    save_folder_path = get_save_folder_path(args)
    os.makedirs(save_folder_path)

    # 가장 먼저 args 저장, namespace 형태를 저장하기는 쉽지 않아 딕셔너리 형태로 바꾸고 json으로 저장
    # with open(경로, 모드) as 이름:
    # os.path.join(파일 경로, 파일 이름)
    with open(os.path.join(save_folder_path, 'args.json'), 'w') as f:
        # device가 class로 저장되서 json 오류 발생. device 삭제
        # 원본 parser롤 지우지 않기 위해 args.__dict__복사
        json_args = args.__dict__.copy()
        del json_args['device']
        # json.dump(내용, 위치), vars(args) or args.__dict__
        json.dump(json_args, f, indent=4) # indent로 이쁘게 조정

    train_loader, test_loader = get_dataloader(args)

    model = get_target_module(args)
    loss = nn.CrossEntropyLoss() 
    optim = Adam(model.parameters(), lr=args.lr)

    # 학습 loop 
    best_acc = 0

    for epoch in range(args.epochs): 
        for idx, (image, target) in enumerate(train_loader): 
            image = image.to(args.device)
            target = target.to(args.device)
            
            out = model(image)
            loss_value = loss(out, target)
            optim.zero_grad() 
            loss_value.backward()
            optim.step()

            if idx % 100 == 0 : 
                print(loss_value.item())
                acc = eval(model, test_loader, args)
                print('accuracy : ', acc)
                # 최고 모델 갱신 시 저장
                if best_acc < acc:
                    best_acc = acc
                    # 구조, foward 값, torch가 지정한 attr을 다 저장하면 너무 큼
                    # state_dict는 layer의 weight와 bias 등을 dict 형태로 저장
                    # torch.save(object, path), object: 저장할 모델 객체, path: 저장할 위치명+파일명
                    torch.save(model.state_dict(),
                               # 새로운 모델이 갱신되면 덮어 씌워져서 이전 모델 확인 불가
                               # 이를 방지하기 위해 epoch와 idx 기록
                               os.path.join(save_folder_path, f'best_model_{epoch}_{idx}.ckpt'))
                    print(f'new best model saved! acc: {acc*100:.2f}')

if __name__ == '__main__': 
    main()