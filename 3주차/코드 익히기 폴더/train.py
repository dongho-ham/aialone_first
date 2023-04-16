# import error 방지를 위한 패키지
import os
import sys
sys.path.append(os.getcwd())

import torch.nn as nn 
from torch.optim import Adam

from utils.getModules import get_dataloader, get_target_module
from utils.parser import parser_args
from utils.evaluation import eval


def main():
    args = parser_args()

    train_loader, test_loader = get_dataloader(args)

    model = get_target_module(args)
    loss = nn.CrossEntropyLoss() 
    optim = Adam(model.parameters(), lr=args.lr)

    # 학습 loop 
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
                print('accuracy : ', eval(model, test_loader, args))


if __name__ == '__main__': 
    main()