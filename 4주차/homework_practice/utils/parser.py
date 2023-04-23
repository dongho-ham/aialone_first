# parser를 저장하는 파일입니다.
import torch
import os
import json
import argparse

# args를 불러오는 함수
def parser_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=500)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument("--img_size", type=int, default=32)

    parser.add_argument("--model_type", type=str, default='lenet', choices=['mlp', 'lenet', 'linear', 'conv', 'incep'])
    
    # 기본 폴더명인 save_folder args와 생성되는 파일명 저장
    parser.add_argument("--save_folder", type=str, default='result')
    return parser.parse_args()

def infer_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str)
    parser.add_argument("--image", type=str)
    
    return parser.parse_args()

def load_trained_args(args):
    with open(os.path.join(args.folder, "args.json"), 'r') as f:
        trained_args = json.load(f)
    trained_args['device'] = args.device
    train_args = argparse.Namespace(**train_args)