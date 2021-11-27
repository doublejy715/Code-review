import jittor as jt
from jittor import Module
from jittor import nn
import networks

# 추가
import jittor.transform as transform
from PIL import Image
import edit.mul
import numpy as np

img_size = 512
transform_image = transform.Compose([
        transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 픽셀을 -1 ~ 1 범위로 normalize 한다.
    ])

def read_img(img):
    img = transform_image(img)
    img = jt.array(img)
    img = img.unsqueeze(0)
    return img

class Combine_Model(nn.Module):
    def name(self):
        return 'Combine_Model'
    
    def initialize(self):
        #The axis of x,y; the size of each part
        # image size 512x512
        # 눈, 코, 입 위치가 고정되어야 한다.
        self.part = {'bg': (0, 0, 512),
                     'eye1': (108, 156, 128),
                     'eye2': (255, 156, 128),
                     'nose': (182, 232, 160),
                     'mouth': (169, 301, 192)}

        self.Sketch_Encoder_Part = {}
        self.Gen_Part = {}
        self.Image_Encoder_Part = {}

        for key in self.part.keys():
            # geometry input image가 sketch image인 경우 사용
            self.Sketch_Encoder_Part[key] = networks.GeometryEncoder(input_nc = 3, output_nc = 3, 
                                                                    ngf = 64, n_downsampling = 4, n_blocks = 1)
            # geometry input image가 real image인 경우 사용
            self.Image_Encoder_Part[key] = networks.GeometryEncoder(input_nc = 3, output_nc = 3, 
                                                                    ngf = 64, n_downsampling = 4, n_blocks = 6)
            # geometry feature와 appearance feature가 합성된 요소 feature map 생성                                                        
            self.Gen_Part[key] = networks.Part_Generator(input_nc=3, output_nc=3, 
                                                                    ngf = 64, n_downsampling = 4, n_blocks = 4)
        # 얼굴 요소들이 합쳐진 feature map에서 합성된 real image를 생성
        self.netG = networks.GlobalGenerator(input_nc = 64, output_nc = 3, 
                                        ngf = 64, n_downsampling = 4, n_blocks = 4)
            
        # 부분마다 model이 있음 가져옴.
        for key in self.part.keys():
            print("load the weight of " + key)
            self.Sketch_Encoder_Part[key].load('./checkpoints/sketch_encoder/sketch_encoder_' + key + '.pkl')
            self.Image_Encoder_Part[key].load('./checkpoints/image_encoder/image_encoder_' + key + '.pkl')
            self.Gen_Part[key].load('./checkpoints/generator/generator_' + key + '.pkl')

        print("load the weight of global fuse")
        self.netG.load('./checkpoints/global_fus.pkl')

    #--------------------------------
    # edited image 생성
    #   1. geometry image와 appearance image에서 요소별로 crop한다. 그리고 요소별로 geometry feature map을 만든다.
    #       1.1. 만약 image가 real image라면 -> Image_Encoder_Part
    #       1.2. 만약 image가 sketch image라면 -> Sketch_Encoder_Part
    #   2. 요소별로 geometry, appearance 정보를 합성한다.
    #   3. background feature map에 요소별로 feature를 붙여넣는다.
    #   4. feature map에서 editied real image를 생성한다.
    #--------------------------------
    def inference(self, args, sketch, appear, geo_type,step=1):
        part_feature = {}

        for key in self.part.keys():
            # 1.
            sketch_part = sketch[:,:,self.part[key][1]: self.part[key][1] + self.part[key][2], self.part[key][0]: self.part[key][0] + self.part[key][2]]
            appear_part = appear[:,:,self.part[key][1]: self.part[key][1] + self.part[key][2], self.part[key][0]: self.part[key][0] + self.part[key][2]]
            with jt.no_grad():
                if geo_type == "sketch":
                    if key == 'mouth' and step != 1:
                        # 여기서 얼굴 수정해 줘야함
                        sketch_part = edit.mul_backup.resize_component(args,sketch_part,key,step)
                        sketch_part = read_img(np.transpose(sketch_part,(2,0,1)))
                    # 1.1.
                    sketch_feature = self.Sketch_Encoder_Part[key](sketch_part)

                else:
                    # 1.2.
                    sketch_feature = self.Image_Encoder_Part[key](sketch_part)
                # 2.
                part_feature[key] = self.Gen_Part[key].feature_execute(sketch_feature, appear_part)
        
        # 3.
        bg_r_feature = part_feature['bg']
        bg_r_feature[:, :, 301:301 + 192, 169:169 + 192] = part_feature['mouth']
        bg_r_feature[:, :, 232:232 + 160 - 36, 182:182 + 160] = part_feature['nose'][:, :, :-36, :]
        bg_r_feature[:, :, 156:156 + 128, 108:108 + 128] = part_feature['eye1']
        bg_r_feature[:, :, 156:156 + 128, 255:255 + 128] = part_feature['eye2']    
        # 전체 feature map 완성!!

        # 4.
        with jt.no_grad():
            fake_image = self.netG(bg_r_feature)

        return fake_image



