"""SeFa."""

import os
import argparse
from tqdm import tqdm
import numpy as np

import torch

from models import parse_gan_type
from utils import to_tensor
from utils import postprocess
from utils import load_generator
from utils import factorize_weight
from utils import HtmlPageVisualizer

# add
from PIL import Image
import torchvision
from torchvision import transforms
import imageio


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Discover semantics from the pre-trained weight.')
    parser.add_argument('--model_name', type=str, default='stylegan_animeface512',
                        help='Name to the pre-trained model.')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save the visualization pages. '
                             '(default: %(default)s)')
    parser.add_argument('-L', '--layer_idx', type=str, default='5-10',
                        help='Indices of layers to interpret. '
                             '(default: %(default)s)')
    parser.add_argument('-N', '--num_samples', type=int, default=5,
                        help='Number of samples used for visualization. '
                             '(default: %(default)s)')
    parser.add_argument('-K', '--num_semantics', type=int, default=5,
                        help='Number of semantic boundaries corresponding to '
                             'the top-k eigen values. (default: %(default)s)')
    parser.add_argument('--start_distance', type=float, default=-4.0,
                        help='Start point for manipulation on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--end_distance', type=float, default=4.0,
                        help='Ending point for manipulation on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--step', type=int, default=11,
                        help='Manipulation step on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--viz_size', type=int, default=256,
                        help='Size of images to visualize on the HTML page. '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_psi', type=float, default=0.7,
                        help='Psi factor used for truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_layers', type=int, default=8,
                        help='Number of layers to perform truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--seed', type=int, default=69,
                        help='Seed for sampling. (default: %(default)s)')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='GPU(s) to use. (default: %(default)s)')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.makedirs(args.save_dir, exist_ok=True)

    # Factorize weights.
    generator = load_generator(args.model_name)
    
    #gan_type = parse_gan_type(generator)
    layers, boundaries, values = factorize_weight(generator, args.layer_idx)

    # Set random seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Prepare codes.
    code = torch.randn(1, 512).cuda() # ?????? ?????? x z dim ????????? matrix??? ??????
    code = generator.mapping(code,None)
    code = code.detach().cpu().numpy()

    # Generate visualization pages.
    distances = np.linspace(args.start_distance,args.end_distance, args.step) # -3 ~ +3 ?????? step ???????????? ?????? ??????
    # ????????? semantic??? ???????????? ??????
    num_sem = args.num_semantics

    for sem_id in tqdm(range(num_sem), desc='Semantic ', leave=False):
          boundary = boundaries[sem_id:sem_id + 1]
          for col_id, d in enumerate(distances, start=1):
               os.makedirs(f'results/{sem_id}',exist_ok=True)
               temp_code = code.copy()
               temp_code[:, layers, :] += boundary * d # temp_code(=code)??? ?????? layer??? broundary * step?????? ?????????. w vector ??????
               image = generator.synthesis(to_tensor(temp_code),noise_mode='const') # ????????? ????????? synethesis??? input??? 
               
               if col_id == 1:
                    image = postprocess(image)[0] # ????????? ?????????
                    tmp = image
               else:
                    images = postprocess(image)[0] # ????????? ?????????
                    tmp = np.concatenate((tmp,images),axis=1)
          imageio.imwrite(f'results/{sem_id}.png',tmp)


if __name__ == '__main__':
    main()
