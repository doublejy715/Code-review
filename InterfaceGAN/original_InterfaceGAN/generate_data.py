# python3.7
"""
generate_data_custom.py
  latent space에서 랜덤으로 샘플링하여 n개의 이미지를 만들어내는 python file

  Input
    - input model(-m) : model .pth file
    - output dir(-o)
    - number(-n)

  output
    - model의 latent space에서 추출된 n 개의 이미지
"""

import os.path
import argparse
from collections import defaultdict
import cv2
import numpy as np
from tqdm import tqdm

from models.model_settings import MODEL_POOL
from models.pggan_generator import PGGANGenerator
from models.stylegan_generator import StyleGANGenerator
from utils.logger import setup_logger

#--------------------------------
# get parser info from input
#--------------------------------
def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Generate images with given model.')
  parser.add_argument('-m', '--model_name', type=str, required=False, default='pggan_celebahq',
                      choices=list(MODEL_POOL),
                      help='Name of the model for generation. (required)')
  parser.add_argument('-o', '--output_dir', type=str, required=False, default='data/pggan_celebahq',
                      help='Directory to save the output results. (required)')
  parser.add_argument('-i', '--latent_codes_path', type=str, default='',
                      help='If specified, will load latent codes from given '
                           'path instead of randomly sampling. (optional)')
  parser.add_argument('-n', '--num', type=int, default=1,
                      help='Number of images to generate. This field will be '
                           'ignored if `latent_codes_path` is specified. '
                           '(default: 1)')
  parser.add_argument('-s', '--latent_space_type', type=str, default='z',
                      choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                      help='Latent space used in Style GAN. (default: `Z`)')
  parser.add_argument('-S', '--generate_style', action='store_true',
                      help='If specified, will generate layer-wise style codes '
                           'in Style GAN. (default: do not generate styles)')
  parser.add_argument('-I', '--generate_image', action='store_false',
                      help='If specified, will skip generating images in '
                           'Style GAN. (default: generate images)')

  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()

  # 과정을 기록할 loagger
  # txt file로 저장해 준다.
  logger = setup_logger(args.output_dir, logger_name='generate_data')
  logger.info(f'Initializing generator.')

  # 따로 정의한 model 불러오기
  gan_type = MODEL_POOL[args.model_name]['gan_type']
  if gan_type == 'pggan':
    model = PGGANGenerator(args.model_name, logger)
    kwargs = {}
  elif gan_type == 'stylegan':
    model = StyleGANGenerator(args.model_name, logger)
    kwargs = {'latent_space_type': args.latent_space_type} # select latent space type (ex : Z,W,WP)
  else:
    raise NotImplementedError(f'Not implemented GAN type `{gan_type}`!')


  logger.info(f'Preparing latent codes.')
  # latent code 가 있는 경우 바로 styleGAN input으로 만든다.
  # latent code : [1,512] vector
  if os.path.isfile(args.latent_codes_path):
    logger.info(f'  Load latent codes from `{args.latent_codes_path}`.')
    latent_codes = np.load(args.latent_codes_path)
    latent_codes = model.preprocess(latent_codes, **kwargs)
  else:
  # latent code 가 없는 경우 random하게 얻는다.
    logger.info(f'  Sample latent codes randomly.')
    latent_codes = model.easy_sample(args.num, **kwargs)
  total_num = latent_codes.shape[0]


  logger.info(f'Generating {total_num} samples.')
  results = defaultdict(list)
  """
  기존 dict : key와 value가 pair로 항상 필요하였음
  defaultdict : 유사 dict, key만 넣어주면 default value를 자동으로 가짐
  default type : defaultdict(type)를 만들때 매개변수로 자료형을 넣어준다.
  key만 넣어주는 경우 type 자료형 value를 자동으로 가진다.
  """
  pbar = tqdm(total=total_num, leave=False)

  # latent code로 이미지를 만든다.(batch_size)
  for latent_codes_batch in model.get_batch_inputs(latent_codes): # latent_codes.shape = (1,512)
    if gan_type == 'pggan':
      outputs = model.easy_synthesize(latent_codes_batch)
    elif gan_type == 'stylegan':
      outputs = model.easy_synthesize(latent_codes_batch,
                                      **kwargs,
                                      generate_style=args.generate_style,
                                      generate_image=args.generate_image)

    # output으로 dict 형식이 나온다.(image, latent vector)
    # save image
    for key, val in outputs.items(): 
      if key == 'image':
        for image in val:
          save_path = os.path.join(args.output_dir, f'{pbar.n:06d}.jpg')
          cv2.imwrite(save_path, image[:, :, ::-1])
          pbar.update(1)
      else:
        results[key].append(val)
    # 예외시 pbar 처리
    if 'image' not in outputs:
      pbar.update(latent_codes_batch.shape[0])
    if pbar.n % 1000 == 0 or pbar.n == total_num:
      logger.debug(f'  Finish {pbar.n:6d} samples.')
  pbar.close()

  # save output(latent vactor, image) to npy file
  logger.info(f'Saving results.')
  for key, val in results.items():
    save_path = os.path.join(args.output_dir, f'{key}.npy')
    np.save(save_path, np.concatenate(val, axis=0))

if __name__ == '__main__':
  main()
