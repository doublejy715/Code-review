from PIL import Image

import numpy as np
import cv2

import edit.util

# day : 21-11-21
            # x_ratio, y_ratio
part_ratio = {'bg': (1, 1),
            'eye1': (1, 1),
            'eye2': (1, 1),
            'nose': (1, 1),
            'mouth': (1, 1)}

            # right_up_y_point, right_up_x_point, map_size, mid_point_y, mid_point_x
part_size = {'bg': (0, 0, 512, 256, 256),
            'eye1': (108, 156, 128, 108+64, 156+64),
            'eye2': (255, 156, 128, 255+64, 156+64),
            'nose': (182, 232, 160, 182+80, 232+80),
            'mouth': (169, 301, 192, 169+96, 301+96)}

# tensor to image
def tensor_to_numpy(image):
    image = image.squeeze(0).detach().numpy()
    image = (np.transpose(image, (1, 2, 0)) + 1) / 2.0 * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def crop_sketch_image(sketch,ratio,key):
    def check_area():
        up_length, down_length,left_length,right_length = 9999,9999,9999,9999
        # 위로 짧은 경우
        if part_size[key][3]-part_size[key][2]<0:
            up_length = part_size[key][3]
        # 아래로 짧은 경우
        elif part_size[key][3]+part_size[key][2] > sketch.shape[0]:
            down_length = sketch.shape[2] - part_size[key][3]
        # 좌로 짧은 경우
        elif part_size[key][4]-part_size[key][2]<0:
            left_length = part_size[key][4]
        # 우로 짧은 경우
        elif part_size[key][4]+part_size[key][2] > sketch.shape[1]:
            right_length = sketch.shape[1] - part_size[key][4]

        return min(up_length,down_length,left_length,right_length,part_size[key][2])

    half_length = check_area()
    sketch_part = sketch[part_size[key][4]-half_length: part_size[key][4]+half_length, part_size[key][3]-half_length: part_size[key][3]+half_length,:]
    mul_sketch = cv2.resize(sketch_part,None,fx=part_ratio[key][0]*ratio[0],fy=part_ratio[key][1]*ratio[1], interpolation = cv2.INTER_CUBIC)
    cropped_sketch = mul_sketch[mul_sketch.shape[0]//2-part_size[key][2]//2:mul_sketch.shape[0]//2+part_size[key][2]//2,mul_sketch.shape[1]//2-part_size[key][2]//2:mul_sketch.shape[1]//2+part_size[key][2]//2,:]
    
    return cropped_sketch

def resize_component(args, tensor_sketch_image, key, step):
    sketch = tensor_to_numpy(tensor_sketch_image)
    if args.grid_step != 1:
        ratio = round(part_ratio[key][0]+(step-args.grid_step//2)/(args.grid_step*2),2),round(part_ratio[key][1]+(step-args.grid_step//2)/(args.grid_step*2),2)
    else:
        ratio = 1,1

    return crop_sketch_image(sketch,ratio,key)
