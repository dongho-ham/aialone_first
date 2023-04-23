# 평가 코드를 저장하는 폴더입니다.

import torch

def eval(model, loader, args):
        total = 0 
        correct = 0 
        for idx, (image, target) in enumerate(loader):
            image = image.to(args.device)
            target = target.to(args.device)

            out = model(image)
            _, pred = torch.max(out, 1)

            correct += (pred == target).sum().item()
            total += image.shape[0]
        return correct / total 

def eval_class(model, loader, args):
    total = torch.zeros(args.num_classes) 
    correct = torch.zeros(args.num_classes) 
    for i, (image, target) in enumerate(loader):
        image = image.to(args.device)
        target = target.to(args.device)

        out = model(image)
        _, pred = torch.max(out, 1)

        for i in range(args.num_classes): 
            correct[i] += ((target == i) & (pred == i)).sum().item()
            total[i] += (target == i).sum().item()
    return correct, total 