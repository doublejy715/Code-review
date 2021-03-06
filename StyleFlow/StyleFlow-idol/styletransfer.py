"""
# attribute 관련
attr_order = ['Gender', 'Glasses', 'Yaw', 'Pitch', 'Baldness', 'Beard', 'Age', 'Expression']
min_dic = {'Gender': 0, 'Glasses': 0, 'Yaw': -20, 'Pitch': -20, 'Baldness': 0, 'Beard': 0.0, 'Age': 0, 'Expression': 0}
max_dic = {'Gender': 1, 'Glasses': 1, 'Yaw': 20, 'Pitch': 20, 'Baldness': 1, 'Beard': 1, 'Age': 65, 'Expression': 1}

"""
# import
from numpy import float32
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
import cv2



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 7)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.softmax(self.fc5(x),dim=1)
        return x.view(-1,7)

def ss(image):
    print("start extract attribute score!!!")
    resnet50 = models.resnet50(pretrained=True)
    resnet50.to(device)
    resnet50.eval()

    score_model = Model()
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

# score 구하는 모델 들고와야 함 밑 함수 input으로 넣어줘야 함
def score(seed, styleGAN):
    """
    여기서 시작 벡터 값을 조절하면 됨
    """
    seed = 20211018
    all_z = np.array([np.random.RandomState(seed).randn(styleGAN.z_dim)])
    w_vector = styleGAN.mapping(torch.from_numpy(all_z).to(device), None,truncation_psi=0.7)
    # w_vector = torch.from_numpy(np.load('projected_w.npz','r')['w']).to(device)
    # mapping SEEDs -> w latent vector
    start_image = styleGAN.synthesis(w_vector,noise_mode='const')
    start_image = (start_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze(0).cpu().numpy()

    Image.fromarray(start_image).save('images/start.png')

    light_score = np.array(ss(start_image))
    attri_score = face_attri_extracter.MS_Face(start_image)

    return  light_score, attri_score, w_vector.detach().cpu().numpy() # w_vector shape : 1 18 512
# 여기는 random seed를 넣어주면 w latent vector, attr_score, light_score를 가지게 해야함
def update_GT_scene_image(w,attri_score,light_score,model,CNFs, pre_lighting):
    # start synthesis image 정보 저장
    w_current = w
    attr_current = attri_score

    light_current = light_score


    # tmp
    tmp_light = np.array([[[0.],[0,]]])
    light_current = np.concatenate([light_current,np.array(tmp_light)],axis=1)
    #---------------
    attr_current_list = [attr_current[0][i][0] for i in range(len(attr_order))] # 
    # light_current_list = [0 for i in range(7)]
    light_current_list = [0 for i in range(9)]
    array_source = torch.from_numpy(attr_current).type(torch.FloatTensor).cuda()
    # bar의 초기값을 가져와야 함 <- 굳이?? 바로 써도 될 듯

    q_array = torch.from_numpy(w_current).cuda().clone().detach()
    #array_source = torch.from_numpy(array_source).type(torch.FloatTensor).cuda()
    # array_light = torch.from_numpy(light_current).type(torch.FloatTensor).cuda()
    # pre_lighting_distance = [pre_lighting[i] - array_light for i in range(len(lighting_order))]

    # final_array_source = torch.cat([array_light, array_source], dim=1) # 1 7 1  / 1 8 1
    final_array_source = array_source
    # final_array_target = torch.cat([array_light, array_source], dim=1)
    final_array_target = torch.from_numpy(np.array(attr_current,dtype=float32)).cuda()
    fws = CNFs(q_array, final_array_source, zero_padding)


    GAN_image = styleGAN.synthesis(torch.tensor(w_current).to(device),noise_mode='const')
    GAN_image = (GAN_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze(0).cpu().numpy()

    return GAN_image, fws, final_array_target, attr_current_list, q_array, light_current_list,# array_light, pre_lighting_distance

def init_data_points(keep_indexes):
    
    # raw_w = pickle.load(open("data/sg2latents.pickle", "rb"))
    # raw_attr = np.load('data/attributes.npy')
    raw_lights2 = np.load('data/light.npy')
    raw_lights = raw_lights2

    # 필요하진 않을 듯
    # all_attr : 해당 좌표 사진의 attribute 값을 의미하는 것 -> 나중에 azure 붙이면 될 듯?
    # lights 도 마찬가지로 light score로 하면 될 듯


    # all_w = np.array(raw_w['Latent'])[keep_indexes]
    # all_attr = raw_attr[keep_indexes]
    all_lights = raw_lights[keep_indexes]

    # 각 인덱스에 해당하는 light 정보를 가져온다. 조사
    # 6명의 빛 값을 가져온다.
    light0 = torch.from_numpy(raw_lights2[8]).type(torch.FloatTensor).cuda()
    light1 = torch.from_numpy(raw_lights2[33]).type(torch.FloatTensor).cuda()
    light2 = torch.from_numpy(raw_lights2[641]).type(torch.FloatTensor).cuda()
    light3 = torch.from_numpy(raw_lights2[547]).type(torch.FloatTensor).cuda()
    light4 = torch.from_numpy(raw_lights2[28]).type(torch.FloatTensor).cuda()
    light5 = torch.from_numpy(raw_lights2[34]).type(torch.FloatTensor).cuda()

    # shape : 1 6 1 9 1 1
    pre_lighting = [light0, light1, light2, light3, light4, light5]

    return pre_lighting

def init_deep_model(opt):
    with dnnlib.util.open_url('model/idol_2K_800K.pkl') as f:
    #with dnnlib.util.open_url('model/ffhq.pkl') as f:
        styleGAN = legacy.load_network_pkl(f)['G_ema']

    generator = styleGAN.cuda()
    w_avg = styleGAN.mapping.w_avg

    generator.eval()
    prior = cnf(512, '512-512-512-512-512', 8, 1) # 수정 필요
    prior.load_state_dict(torch.load('trained_model/modellarge10k_001000_02.pt')) # 
    prior.eval()

    return styleGAN, w_avg, prior

# init_deep_model
def init(opt, keep_indexes):
    keep_indexes = np.array(keep_indexes).astype(np.int)
    return init_deep_model(opt)


#--------------------------------------
# Start!
#--------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda")

    opt = TestOptions().parse()

    truncation_psi = 0.5
    keep_indexes = [2, 5, 25, 28, 16, 32, 33, 34, 55, 75, 79, 162, 177, 196, 160, 212, 246, 285, 300, 329, 362,
                                369, 462, 460, 478, 551, 583, 643, 879, 852, 914, 999, 976, 627, 844, 237, 52, 301,
                                599]
    attr_order = ['Gender', 'Glasses', 'Yaw', 'Pitch', 'Baldness', 'Beard', 'Age', 'Expression']
    # attr_order = ['Yaw']
    lighting_order = ['Left->Right', 'Right->Left', 'Down->Up', 'Up->Down', 'No light', 'Front light']

    zero_padding = torch.zeros(1, 18, 1).cuda()

    seeds = None


    """
    # attribute 관련
    attr_order = ['Gender', 'Glasses', 'Yaw', 'Pitch', 'Baldness', 'Beard', 'Age', 'Expression']
    min_dic = {'Gender': 0, 'Glasses': 0, 'Yaw': -20, 'Pitch': -20, 'Baldness': 0, 'Beard': 0.0, 'Age': 0, 'Expression': 0}
    max_dic = {'Gender': 1, 'Glasses': 1, 'Yaw': 20, 'Pitch': 20, 'Baldness': 1, 'Beard': 1, 'Age': 65, 'Expression': 1}
    """

    # 값이 들어가야 한다.
    want_slide_value = 2
    attri_index = 2

    raw_slide_value_light = 0.5
    light_index = 0

    styleGAN, w_avg, CNFs = init(opt,keep_indexes)
    pre_lighting = init_data_points(keep_indexes)
    light_score, attri_score, w_vector = score(seeds,styleGAN) # attri_score, light_score : 처음 이미지의 score
    #GAN_image, fws, final_array_target, attr_current_list, q_array,light_current_list, array_light, pre_lighting_distance = update_GT_scene_image(w_vector,attri_score,light_score,styleGAN,CNFs,pre_lighting)
    GAN_image, fws, final_array_target, attr_current_list, q_array,light_current_list = update_GT_scene_image(w_vector,attri_score,light_score,styleGAN,CNFs,pre_lighting)
    # editing 조건을 걸어줄 것

    # 이거 매개변수 더 찾아봐야 할 듯??
    # 몇번째 index를 변화시키는지 확인하는것 같은데.... attribute 
    # attri_index : 아마 변한 수치의 인덱스 (int)


    """
    i수치를 위에 주석범위 사이로 설정하면 interpolation grid 이미지 얻기 가능
    """
    os.makedirs("transfer_edit_result", exist_ok=True)


    for i in [-20,20]:
        attri_index = 2
        edit_image, q_array, fws = real_time_arrti(attri_index,i,final_array_target,zero_padding, fws, CNFs, styleGAN, attr_current_list, q_array) # attri_index : / raw_slide_value_attri : 바뀌는 attri score 임
        for j in [-20,20]:
            attri_index = 3
            edit_image, q_array, fws = real_time_arrti(attri_index,j,final_array_target,zero_padding, fws, CNFs, styleGAN, attr_current_list, q_array) # attri_index : / raw_slide_value_attri : 바뀌는 attri score 임
            for k in [0,1]:
                attri_index = 7
                edit_image, q_array, fws = real_time_arrti(attri_index,k,final_array_target,zero_padding, fws, CNFs, styleGAN, attr_current_list, q_array) # attri_index : / raw_slide_value_attri : 바뀌는 attri score 임
                tmp = GAN_image
                ##############
                # 스타일 적용 #
                ##############
                for l in range(10):
                    # 스타일 제공해줄 W 만들기 (이미지를 만들거나 저장할 필요는 X)
                    style_seed = np.random.randint(1, 1000)
                    style_z = np.random.RandomState(style_seed).randn(styleGAN.z_dim)
                    style_w = styleGAN.mapping(torch.from_numpy(style_z).unsqueeze(0).to(device), None, truncation_psi=0.7)
                    style_img = styleGAN.synthesis(style_w, noise_mode="const")

                    # W 중 일부 스타일 넘겨 받아서 만들기
                    target_w_temp = q_array.clone()
                    target_w_temp[:, 7:, :] = style_w[:, 7:, :]
                    target_style_img = styleGAN.synthesis(target_w_temp, noise_mode="const")

                    # 저장하기 위해 필요한 부분
                    style_img = (style_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
                    target_style_img = (target_style_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]

                    tmp = np.concatenate((tmp,edit_image),axis=1)
                imageio.imwrite(f'results/yaw{i}_pitch{j}_expres{k}.png',tmp)

    print("JOB FINISH!!")
    # edit_image = real_time_lighting(light_index, raw_slide_value_light,light_current_list, array_light, final_array_target, fws, CNFs, styleGAN, q_array, zero_padding, pre_lighting_distance) # 이하 동문

    # 마지막으로 결과물 이미지 save

