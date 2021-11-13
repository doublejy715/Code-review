import jittor as jt
from jittor import Module
from jittor import nn
import numpy as np
import jittor.transform as transform
from PIL import Image
from combine_model import Combine_Model
import networks
from argparse import ArgumentParser


img_size = 512
transform_image = transform.Compose([
        transform.Resize(size = img_size),
        transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 픽셀을 -1 ~ 1 범위로 normalize 한다.
    ])

def read_img(path):
    img = Image.open(path).convert('RGB')
    img = transform_image(img)
    img = jt.array(img)
    img = img.unsqueeze(0)
    return img

def save_img(image, path):
    # output 도 -1 ~ 1 범위이므로 0 ~ 255 로 바꿔준다.
    # 순서는 HxWxC -> CxHxW 
    # -1 ~ 1 -> 0 ~ 1 -> *255
    image = image.squeeze(0).detach().numpy()
    image = (np.transpose(image, (1, 2, 0)) + 1) / 2.0 * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)

if __name__ == '__main__':
    #---------------------------------
    # parser
    #---------------------------------
    parser = ArgumentParser()
    parser.add_argument("--geo", type=str, default = "./images/17115_sketch.png", help = "the path of geometry image")
    parser.add_argument("--appear", type=str, default = "./images/69451.png", help = "the path of appearance image")
    parser.add_argument("--output", type=str, default = "./results/sketch_result.png", help = "the path of output image")
    parser.add_argument("--cuda", type=int, default = 1, help = "use cuda or cpu: 0 , cpu; 1 , gpu")
    parser.add_argument("--geo_type", type=str, default="sketch", help = "extract geometry from image or sketch: sketch / image")
    parser.add_argument("--gen_sketch", action='store_true', help = "with --gen_sketch, extract sketch from real image")
    args = parser.parse_args()

    jt.flags.use_cuda = args.cuda

    # model and save image
    if args.gen_sketch:
        #---------------------------------
        # real image to sketch image
        #   geometry 정보를 사용하고 싶은 image가 color일 때 sketch image를 생성하기 위해 사용
        #   input : real image
        #   output : sketch image
        #---------------------------------
        sketch_netG = networks.GlobalGenerator(input_nc = 3, output_nc = 3, 
                                        ngf = 32, n_downsampling = 4, n_blocks = 9)
        sketch_netG.load("./checkpoints/sketch_generator.pkl")
        geo_img = read_img(args.geo)
        with jt.no_grad():
            sketch = sketch_netG(geo_img)
            save_img(sketch, args.output)
    else:
        #---------------------------------
        # geometry와 appearance를 합치는 과정
        #   input : appearance image and geometry image(real image or sketch image)
        #   output : editted image
        #---------------------------------
        # image preprocessing
        geo_img = read_img(args.geo)
        appear_img = read_img(args.appear)

        # create model & load pkl file
        #   GeometryEncoder, Part_Generator, GlobalGenerator
        model = Combine_Model()
        model.initialize()

        # image generate 시작
        geo_type = args.geo_type
        image_swap = model.inference(geo_img, appear_img, geo_type)
        save_img(image_swap, args.output)



