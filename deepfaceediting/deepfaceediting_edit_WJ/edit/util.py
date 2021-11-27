import cv2
import numpy as np

# ratio[0]=x, ratio[1]=y
def crop_sketch_image(image,ratio,key):
    drawing_paper = np.ones_like(image)*255
    
    hight,width = image.shape[:2]


    mul_image = cv2.resize(image,(round(ratio[1]*hight),round(ratio[0]*width)), interpolation = cv2.INTER_CUBIC)
    mul_mid_w,mul_mid_h,original_half_size = mul_image.shape[1]//2 if mul_image.shape[1]%2==0 else mul_image.shape[1]//2+1\
        ,mul_image.shape[0]//2 if mul_image.shape[0]%2==0 else mul_image.shape[0]//2+1 \
            , part_ratio[key][2]//2

    # 0.xx 배율로 resize할 때 0.xx * image.size 시 소수점이 되버려서 drawing paper에 삽입이 안됨

    if ratio[0] >= 1 and ratio[1] < 1:
        mul_half_size = mul_image.shape[0]//2
        crop_image = mul_image[:,mul_half_size-original_half_size:mul_half_size+original_half_size,:]
        h,w,mid = crop_image.shape[0]//2,crop_image.shape[1]//2,image.shape[0]//2
        drawing_paper[mid-h:mid+h,mid-w:mid+w,:] = crop_image
        return drawing_paper

    elif ratio[0] < 1 and ratio[1] >= 1:
        mul_half_size = mul_image.shape[1]//2
        crop_image = mul_image[mul_half_size-original_half_size:mul_half_size+original_half_size,:,:]
        h,w,mid = crop_image.shape[0]//2,crop_image.shape[1]//2,image.shape[0]//2
        drawing_paper[mid-h:mid+h,mid-w:mid+w,:] = crop_image
        return drawing_paper

    elif ratio[0] < 1 and ratio[1] < 1:
        mul_image
        drawing_paper[original_half_size-mul_mid_w:original_half_size+mul_mid_w,original_half_size-mul_mid_h:original_half_size+mul_mid_h,:] = mul_image
        return drawing_paper

    else:
        return mul_image[mul_mid_w-original_half_size:mul_mid_w+original_half_size,mul_mid_h-original_half_size:mul_mid_h+original_half_size,:]
