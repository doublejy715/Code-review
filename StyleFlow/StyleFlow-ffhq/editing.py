import torch

device = torch.device("cuda")

square_size = 100

attr_degree_list = [1.5, 2.5, 1., 1., 2, 1.7,0.93, 1.]

attr_order = ['Gender', 'Glasses', 'Yaw', 'Pitch', 'Baldness', 'Beard', 'Age', 'Expression']
lighting_order = ['Left->Right', 'Right->Left', 'Down->Up', 'Up->Down', 'No light', 'Front light']

# attribute 관련
min_dic = {'Gender': 0, 'Glasses': 0, 'Yaw': -20, 'Pitch': -20, 'Baldness': 0, 'Beard': 0.0, 'Age': 0, 'Expression': 0}
max_dic = {'Gender': 1, 'Glasses': 1, 'Yaw': 20, 'Pitch': 20, 'Baldness': 1, 'Beard': 1, 'Age': 65, 'Expression': 1}
attr_interval = 80
interval_dic = {'Gender': attr_interval, 'Glasses': attr_interval, 'Yaw': attr_interval, 'Pitch': attr_interval,
                'Baldness': attr_interval, 'Beard': attr_interval, 'Age': attr_interval, 'Expression': attr_interval}
# set_values_dic = {i: int(interval_dic[i]/2) for i in interval_dic}
gap_dic = {i: max_dic[i] - min_dic[i] for i in max_dic}


# light 관련
light_degree = 1.
light_min_dic = {'Left->Right': 0, 'Right->Left': 0, 'Down->Up': 0, 'Up->Down': 0, 'No light': 0, 'Front light': 0}
light_max_dic = {'Left->Right': light_degree, 'Right->Left': light_degree, 'Down->Up': light_degree, 'Up->Down': light_degree, 'No light': light_degree, 'Front light': light_degree}


light_interval = 80
light_interval_dic = {'Left->Right': light_interval, 'Right->Left': light_interval, 'Down->Up': light_interval,
                      'Up->Down': light_interval, 'No light': light_interval, 'Front light': light_interval}

light_set_values_dic = {i: 0 for i in light_interval_dic}
light_gap_dic = {i: light_max_dic[i] - light_min_dic[i] for i in light_max_dic}


# 변수

def invert_slide_to_real(name, slide_value):
    return float(slide_value /interval_dic[name] * (gap_dic[name]) + min_dic[name])

def light_invert_slide_to_real(name, slide_value):
    return float(slide_value /light_interval_dic[name] * (light_gap_dic[name]) + light_min_dic[name])

# attribute editing
# final_array_target : attri + lighting 스코어 값인듯
# q_array 필요
# 
# attr_index ; [0,1,2,3,4,5,6] -> 아마도 attribute len 일듯 -> 아마도 변한 값의 index값일 듯
# raw_slide_value : 원하는 값을 가진 want attribute score 
# prior : fws model 
# fws : CNF 결과
# zero_padding 


# model 
# def real_time_arrti(attr_index, raw_slide_value, final_array_target, zero_padding, prior,model):
def real_time_arrti(attr_index, edit_slide_value, ori_whole_score, zero_padding, fws, CNFs, styleGAN, attr_current_list, q_array):
    # real_value : 실제 변화를 원하는 값인데, 범위를 지정해 줘야 할듯 parser에서
    # 먼저 real_value의 정보들을 담은 list 가 필요
    # 이부분은 임의로 지정한다.
    ori_whole_score = ori_whole_score.unsqueeze(-1)
    
    real_value = invert_slide_to_real(attr_order[attr_index], edit_slide_value)

    attr_change = real_value - attr_current_list[0]
    attr_final = attr_degree_list[0] * attr_change + attr_current_list[0]

    #ori_whole_score[0, 0 + 9, 0, 0] = attr_final
    ori_whole_score[0, 0 , 0, 0] = attr_final

    rev = CNFs(fws[0], ori_whole_score, zero_padding, True)

    if attr_index == 0:
        rev[0][0][8:] = q_array[0][8:]

    elif attr_index == 1:
        rev[0][0][:2] = q_array[0][:2]
        rev[0][0][4:] = q_array[0][4:]

    elif attr_index == 2:

        rev[0][0][4:] = q_array[0][4:]

    elif attr_index == 3:
        rev[0][0][4:] = q_array[0][4:]

    elif attr_index == 4:
        rev[0][0][6:] = q_array[0][6:]

    elif attr_index == 5:
        rev[0][0][:5] = q_array[0][:5]
        rev[0][0][10:] = q_array[0][10:]

    elif attr_index == 6:
        rev[0][0][0:4] = q_array[0][0:4]
        rev[0][0][8:] = q_array[0][8:]

    elif attr_index == 7:
        rev[0][0][:4] = q_array[0][:4]
        rev[0][0][6:] = q_array[0][6:]

    w_current = rev[0].detach().cpu().numpy()
    q_array = torch.from_numpy(w_current).cuda().clone().detach()

    fws = CNFs(q_array, ori_whole_score, zero_padding)

    GAN_image = styleGAN.synthesis(torch.tensor(w_current).to(device),noise_mode='const')
    GAN_image = (GAN_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze(0).cpu().numpy()

    return GAN_image, q_array, fws

#light_editing
def real_time_lighting(light_index, raw_slide_value,light_current_list, array_light,final_array_target, fws,CNFs, styleGAN, q_array, zero_padding, pre_lighting_distance):
    
    real_value = light_invert_slide_to_real(lighting_order[light_index], raw_slide_value)

    light_current_list[light_index] = real_value

    ###############################
    ###############  calculate attributes array first, then change the values of attributes

    lighting_final = array_light.clone().detach()
    for i in range(len(lighting_order)):
        lighting_final += light_current_list[i] * pre_lighting_distance[i]

    final_array_target[:, :9] = lighting_final

    rev = CNFs(fws[0], final_array_target, zero_padding, True)
    rev[0][0][0:7] = q_array[0][0:7]
    rev[0][0][12:18] = q_array[0][12:18]

    w_current = rev[0].detach().cpu().numpy()
    q_array = torch.from_numpy(w_current).cuda().clone().detach()

    fws = CNFs(q_array, final_array_target, zero_padding)

    GAN_image = styleGAN.generate_im_from_w_space(w_current)[0]   

    return GAN_image