# import
from options.test_options import TestOptions
import numpy as np
import torch
import pickle
import torch
# 필요 파일들
# from utils import Build_model
from module.flow import cnf
from editing import *
import os


import face_attri_extracter
import light_score 

import dnnlib
import legacy
from PIL import Image
import imageio

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import numpy as np

device = torch.device("cuda")


def ss(image):
    print("start extract attribute score!!!")
    resnet50 = models.resnet50(pretrained=True)
    resnet50.to(device)
    resnet50.eval()
    
    score_model = light_score.Model()
    score_model.load_state_dict(torch.load('light_score_2.pt'))
    score_model.to(device)
    score_model.eval()

    img = Image.fromarray(image)

    img = img.resize((512,512))
    trans = transforms.ToTensor()
    img = trans(img).float().to(device)
    
    outputs = resnet50(img.unsqueeze(0))
    outputs = score_model(outputs).detach().cpu()
    outputs = outputs.unsqueeze(-1).numpy()
    
    print("END extract attribute score!!!")

    return outputs

def score(seed, styleGAN):
    # 여기서 seed를 넣을 것인지 projection으로 생성할 것인지 코드 추가
    all_z = np.array([np.random.RandomState(seed).randn(styleGAN.z_dim)])
    w_vector = styleGAN.mapping(torch.from_numpy(all_z).to(device), None)
    # mapping SEEDs -> w latent vector
    start_image = styleGAN.synthesis(w_vector,noise_mode='const')
    start_image = (start_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze(0).cpu().numpy()

    Image.fromarray(start_image).save('images/start.png')

    light_score = np.array(ss(start_image))
    attri_score = face_attri_extracter.MS_Face(start_image)

    return  light_score, attri_score, w_vector.detach().cpu().numpy() # w_vector shape : 1 18 512
# 여기는 random seed를 넣어주면 w latent vector, attr_score, light_score를 가지게 해야함
def prepareing(w,attri_score,light_score,model,CNFs, pre_lighting):
    # start synthesis image 정보 저장
    w_current = w
    attr_current = attri_score

    light_current = light_score

    attr_current_list = [attr_current[0][i][0] for i in range(len(attr_order))] # 
    # light_current_list = [0 for i in range(7)]
    light_current_list = [0 for i in range(7)]

    q_array = torch.from_numpy(w_current).cuda().clone().detach()
    array_source = torch.from_numpy(attr_current).type(torch.FloatTensor).cuda()
    array_light = torch.from_numpy(light_current).type(torch.FloatTensor).cuda()
    pre_lighting_distance = [pre_lighting[i] - array_light for i in range(len(lighting_order))]

    final_array_source = torch.cat([array_light, array_source], dim=1)
    final_array_target = torch.cat([array_light, array_source], dim=1)

    fws = CNFs(q_array, final_array_source, zero_padding)

    GAN_image = styleGAN.synthesis(torch.tensor(w_vector).to(device),noise_mode='const')
    GAN_image = (GAN_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze(0).cpu().numpy()

    return GAN_image, fws, final_array_target, attr_current_list, q_array, light_current_list, array_light, pre_lighting_distance

def init_data_points(keep_indexes):
    raw_lights2 = np.load('data/light.npy')
    raw_lights = raw_lights2

    all_lights = raw_lights[keep_indexes]

    light0 = torch.from_numpy(raw_lights2[8]).type(torch.FloatTensor).cuda()
    light1 = torch.from_numpy(raw_lights2[33]).type(torch.FloatTensor).cuda()
    light2 = torch.from_numpy(raw_lights2[641]).type(torch.FloatTensor).cuda()
    light3 = torch.from_numpy(raw_lights2[547]).type(torch.FloatTensor).cuda()
    light4 = torch.from_numpy(raw_lights2[28]).type(torch.FloatTensor).cuda()
    light5 = torch.from_numpy(raw_lights2[34]).type(torch.FloatTensor).cuda()

    pre_lighting = [light0, light1, light2, light3, light4, light5]

    return pre_lighting

def init_deep_model(opt):
    with dnnlib.util.open_url('model/network-snapshot-002800.pkl') as f:
        styleGAN = legacy.load_network_pkl(f)['G_ema']

    generator = styleGAN.cuda()
    w_avg = styleGAN.mapping.w_avg

    generator.eval()
    prior = cnf(512, '512-512-512-512-512', 15, 1)
    prior.load_state_dict(torch.load('traind_model/modellarge10k_001000_02.pt')) # 
    prior.eval()

    return styleGAN, w_avg, prior

# init_deep_model
def init(opt, keep_indexes):
    keep_indexes = np.array(keep_indexes).astype(np.int)
    return init_deep_model(opt)


#--------------------------------------
# Start!
#--------------------------------------
opt = TestOptions().parse()

truncation_psi = 0.5

keep_indexes = [2, 5, 25, 28, 16, 32, 33, 34, 55, 75, 79, 162, 177, 196, 160, 212, 246, 285, 300, 329, 362,
                            369, 462, 460, 478, 551, 583, 643, 879, 852, 914, 999, 976, 627, 844, 237, 52, 301,
                            599]

attr_order = ['Gender', 'Glasses', 'Yaw', 'Pitch', 'Baldness', 'Beard', 'Age', 'Expression']
lighting_order = ['Left->Right', 'Right->Left', 'Down->Up', 'Up->Down', 'No light', 'Front light']

zero_padding = torch.zeros(1, 16, 1).cuda()

seeds = None

# parser
# input -> seed / image 
# 원하는 attribuet 지정
# 얼마나 변화시킬지 지정
# output -> dir 위치 지정 / gird or single image



# 여기서 원하는 attribute index 선택 / semantic value 선택
want_slide_value = 0
attri_index = 1

styleGAN, w_avg, CNFs = init(opt,keep_indexes)
pre_lighting = init_data_points(keep_indexes)
light_score, attri_score, w_vector = score(seeds,styleGAN)
GAN_image, fws, final_array_target, attr_current_list, q_array,light_current_list, array_light, pre_lighting_distance = prepareing(w_vector,attri_score,light_score,styleGAN,CNFs,pre_lighting)

tmp = GAN_image

# 여기서 수치 범위를 정해주면 됨
for i in [-19,-14,-7,0,7,14,20]:
    edit_image, q_array, fws = real_time_arrti(attri_index,i,final_array_target,zero_padding, fws, CNFs, styleGAN, attr_current_list, q_array) # attri_index : / raw_slide_value_attri : 바뀌는 attri score 임
    tmp = np.concatenate((tmp,edit_image),axis=1)
imageio.imwrite(f'results/finish.png',tmp)

# 마지막으로 결과물 이미지 save