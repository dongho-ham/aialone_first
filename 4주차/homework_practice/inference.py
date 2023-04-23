# 패키지 불러오기
from utils.parser import infer_parser_args
from utils.getModules import get_target_module, get_Transform
from PIL import Image
import torch.nn.functional as F
import os
import torch


def main():
    args = infer_parser_args()

    assert os.path.exits(args.folder)
    assert os.path.exits(args.image)

    model = get_target_module(args)

    model.load_state_dict(torch.load(os.path.join(args.folder, 'best_model.ckpt')))

    transform = get_Transform(args)

    img = Image.open(args.image)
    img = transform(img)
    img = img.unsqueeze(0)
    
    output = model(img)

    prob = F.softmax(output, 1)
    index = torch.argmax(prob)
    score = torch.max(prob)
    

if __name__ == '__main__':
    main()