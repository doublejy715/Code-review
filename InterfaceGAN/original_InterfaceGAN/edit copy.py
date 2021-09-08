# python3.7
"""
Interpolation file
이미지에서 특성 하나를 변화시켜주는 파일

  intput
    - model
    - boundary(select attribute)
    - latent code npy
  output
    - eddited image

  어떤 과정으로 이미지 생성하는지 확인
"""

import os.path
import argparse
import cv2
import numpy as np
from tqdm import tqdm

from models.model_settings import MODEL_POOL
from models.pggan_generator import PGGANGenerator
from models.stylegan_generator import StyleGANGenerator
from utils.logger import setup_logger
from utils.manipulator import linear_interpolate


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Edit image synthesis with given semantic boundary.')
  parser.add_argument('-m', '--model_name', type=str, required=True,
                      choices=list(MODEL_POOL),
                      help='Name of the model for generation. (required)')
  parser.add_argument('-o', '--output_dir', type=str, required=True,
                      help='Directory to save the output results. (required)')
  parser.add_argument('-b', '--boundary_path', type=str, required=True,
                      help='Path to the semantic boundary. (required)')
  parser.add_argument('-i', '--input_latent_codes_path', type=str, default='',
                      help='If specified, will load latent codes from given '
                           'path instead of randomly sampling. (optional)')
  parser.add_argument('-n', '--num', type=int, default=1,
                      help='Number of images for editing. This field will be '
                           'ignored if `input_latent_codes_path` is specified. '
                           '(default: 1)')
  parser.add_argument('-s', '--latent_space_type', type=str, default='z',
                      choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                      help='Latent space used in Style GAN. (default: `Z`)')
  parser.add_argument('--start_distance', type=float, default=-3.0,
                      help='Start point for manipulation in latent space. '
                           '(default: -3.0)')
  parser.add_argument('--end_distance', type=float, default=3.0,
                      help='End point for manipulation in latent space. '
                           '(default: 3.0)')
  parser.add_argument('--steps', type=int, default=10,
                      help='Number of steps for image editing. (default: 10)')

  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  os.makedirs(args.output_dir,exist_ok = True)
  # pkl file 을 이용하기 위해서는 legacy.py / dnnlib file / torch_utils file 필요
  # 68 ~ 74 line : load model 
  import legacy
  import dnnlib
  import torch

  device = torch.device('cuda:1')
  with dnnlib.util.open_url('models/pretrain/network-snapshot-000500.pkl') as f:
      model = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

  # load boundary
  boundary = np.load(args.boundary_path)

  """
  original edit.py의 과정을 알아내고 필요한 부분만을 가져온다.
  edit.py - manipulator.py - def linear_interpolate 함수
  np.linspace를 이용하여 start latent vector를 이동시킨다.

    1) 상수가 들어가는 변수는 적절하게 수정
    2) 가져온 코드를 이용하기 위해서 어떤 형식, 형태인지 미리 파악한다.
    3) interpolate 외에도 대체 가능한 코드가 있으면 대체한다.
  """
  seed = 777
  steps = 7
  start_distance = -5.0
  end_distance = 5.0
  latent_code = np.random.RandomState(seed).randn(1, 512)
  linspace = np.linspace(start_distance, end_distance, steps)
  if len(latent_code.shape) == 2:
    linspace = linspace - latent_code.dot(boundary.T)
    linspace = linspace.reshape(-1, 1).astype(np.float32)
    latent_code = latent_code + linspace * boundary
  if len(latent_code.shape) == 3:
    linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
    latent_code = latent_code + linspace * boundary.reshape(1, 1, -1)

  print(np.shape(latent_code))
  import PIL
  import torchvision
  from torchvision import transforms

  image_list = []
  for latent in latent_code:
    z = torch.from_numpy(latent).unsqueeze(0).to(device)
    img = model(z, torch.zeros_like(z).to(device), truncation_psi=0.7, noise_mode='const')
    image_list.append(img)
  output = torchvision.utils.make_grid(torch.cat(image_list, dim=0), nrow=steps)
  output = transforms.ToPILImage()(output.cpu().squeeze().clamp(0, 1))
  output.save(f"results/test.png")

if __name__ == '__main__':
  main()
