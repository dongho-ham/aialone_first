# 패키지 불러오기 
import sys 
import os
sys.path.append(os.getcwd())
import json 

import torch 
import torch.nn as nn 
from torch.optim import Adam 

from utils.parser import parser_args 
from utils.getModules import getDataLoader 
from utils.getModules import getTargetModel 
from utils.evaluation import eval 
from misc.tools import get_save_folder_path

def main():
    args = parser_args() 

    # 저장 폴더를 설정 
    save_folder_path = get_save_folder_path(args)
    os.makedirs(save_folder_path)
    with open(os.path.join(save_folder_path, 'args.json'), 'w') as f : 
        json_args = args.__dict__.copy()
        del json_args['device']
        json.dump(json_args, f, indent=4)

    train_loader, test_loader = getDataLoader(args) 

    # 모델, loss, optimizer 
    model = getTargetModel(args) 
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
                if best_acc < acc :
                    best_acc = acc 
                    torch.save(model.state_dict(), 
                               os.path.join(save_folder_path, f'best_model.ckpt'))
                    print(f'new best model saved! acc : {acc*100:.2f}')

if __name__ == '__main__': 
    main()
