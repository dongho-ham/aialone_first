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

    parser.add_argument("--model_type", type=str, default='lenet', choices=['mlp', 'lenet', 'linear', 'conv', 'incep', 'vgg', 'resnet'])
    parser.add_argument("--vgg_type", type=str, default='a', choices=['a','b','c','d','e'])
    parser.add_argument("--res_config", type=str, default='18', choices=['18', '34', '50', '101', '152'])
    
    # 기본 폴더명인 save_folder args와 생성되는 파일명 저장
    parser.add_argument("--save_folder", type=str, default='results')
    return parser.parse_args()

# 추론하는데 있어 기존 parser_args()에서 지정한 parser들은 필요가 없음.
# 새로운 이미지와 저장할 폴더, 그리고 학습에 필요한 device만 있으면 됨.
def infer_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str)
    parser.add_argument("--iamge", type=str)
    parser.add_argument("--device", default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    return parser.parse_args()

# result 폴더에 저장한 args 불러오기
def load_trained_args(args):
    # args.folder 경로와 args.json 파일 이름을 결합해서 경로 생성
    # 해당 경로의 파일을 open 함수로 읽기 전용으로 열고, trained_args에 해당 파일을 json 형식으로 로드해서 저장
    with open(os.path.join(args.folder, "args.json"), 'r') as f:
        trained_args = json.load(f)
    # 이전에 args에 삭제한 device 추가
    trained_args['device'] = args.device
    # **: dict을 언패킹
    trained_args = argparse.Namespace(**trained_args)

    return trained_args