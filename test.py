#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

def evaluate(input_path,result_path,file_name,count,cp='model_final_diss.pth'):
    
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        img = Image.open(input_path + file_name)
        if len(np.array(img)[0][0]) > 3:
            return
        image = img.resize((1024, 1024), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # print(parsing)

        image = np.array(image)
        dys, dxs = np.where(parsing==0)
        for dy, dx in zip(dys,dxs):
            image[dy,dx] = [222,222,222]
        
        result = Image.fromarray(image)
        tmp = str(count).zfill(6)
        result.save(result_path+tmp+'.png')
    print(f"{file_name} done!!!")

if __name__ == "__main__":
    GPU_NUM = 1 # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print(torch.cuda.get_device_name(GPU_NUM))


    input_PATH = 'makeup/'
    result_PATH = 'result/'
    count = 0
    for index in os.listdir('makeup/'):
        evaluate(input_PATH,result_PATH,index,count,cp='79999_iter.pth')
        count += 1

"""
        eye_map = np.zeros_like(parsing)

        image = np.array(image)
        for i in range(image.shape[0]):
          for j in range(image.shape[1]):
                if parsing[i,j] == 5 or parsing[i,j] == 4:
                    eye_map[i][j] = 1
                else:
                    continue
"""