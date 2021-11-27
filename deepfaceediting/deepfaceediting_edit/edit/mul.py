from PIL import Image

import numpy as np
import cv2

import util

# day : 21-11-21
# x_ratio, y_ratio, component_size
part_ratio = {'bg': (1, 1, 512),
                'eye1': (1, 1, 128),
                'eye2': (1, 1, 128),
                'nose': (1, 1, 160),
                'mouth': (1, 1.5, 192)}

# save grid image
def create_grid_image(grid,image):
    grid = np.concatenate((grid,image),axis=1)
    return grid

# tensor to image
def tensor_to_numpy(image):
    image = image.squeeze(0).detach().numpy()
    image = (np.transpose(image, (1, 2, 0)) + 1) / 2.0 * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def crop_sketch_image(image,ratio,key):
    return util.crop_sketch_image(image,ratio,key)

def resize_component(args, tensor_image, key, step):
    ratio = part_ratio[key][0]+(step-args.grid_step//2)/(args.grid_step*2),part_ratio[key][1]+(step-args.grid_step//2)/(args.grid_step*2)

    image = tensor_to_numpy(tensor_image)
    return crop_sketch_image(image,ratio,key)
