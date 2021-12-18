#!/usr/bin/python
# -*- encoding: utf-8 -*-

from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import warnings

from argparse import ArgumentParser

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='../data/seg_alpha.jpg'):
    # Colors for all 20 parts [eye1:10 / eye2 : 14 / nose : 5 / mouse : 7(윗입술) 6(입) 8(아래)] BRG
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    # 
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    
    vis_parsing_anno_result = cv2.resize(vis_parsing_anno, (256,256),interpolation=cv2.INTER_NEAREST)

    cv2.imwrite('../images/segmentation/result.png',vis_parsing_anno_result)

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def evaluate(args,image,cp='model_final_diss.pth'):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    # save_pth = osp.join('face_parsing/res/cp', cp)
    # net.load_state_dict(torch.load(save_pth,map_location='cpu'))
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        #try:
        image = image.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # parsing = cv2.imwrite('../data/seg_result.jpg',parsing)
        vis_parsing_maps(image, parsing, stride=1, save_im=False, save_path=args.input)
        # except:
        #     warnings.warn("deprecated", DeprecationWarning)
    # print(f"{file_name} done!!!")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--cuda", type=int, default = 0, help = "use cuda or cpu: 0 , cpu; 1 , gpu")
    parser.add_argument("--input", type=str, default = "../images/original.png", help = "the path of image")
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    image = Image.open(args.input).convert('RGB')
    torch.cuda.set_device(device)
    evaluate(args,image,cp='79999_iter.pth')