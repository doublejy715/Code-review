# python3.7
"""
  input
    - latent vector z numpy file
    - attribute score numpy file

  output
    - attribute boundary numpy file

  attribute 별로 boundary를 학습한다. 어떻게 input file을 준비할지 알아내야 한다.

"""

import os.path
import argparse
import numpy as np

from utils.logger import setup_logger
from utils.manipulator import train_boundary

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Train semantic boundary with given latent codes and '
                  'attribute scores.')
  parser.add_argument('-o', '--output_dir', type=str, required=True,
                      help='Directory to save the output results. (required)')
  parser.add_argument('-c', '--latent_codes_path', type=str, required=True,
                      help='Path to the input latent codes. (required)')
  parser.add_argument('-s', '--scores_path', type=str, required=True,
                      help='Path to the input attribute scores. (required)')
  parser.add_argument('-n', '--chosen_num_or_ratio', type=float, default=0.02,
                      help='How many samples to choose for training. '
                           '(default: 0.2)')
  parser.add_argument('-r', '--split_ratio', type=float, default=0.7,
                      help='Ratio with which to split training and validation '
                           'sets. (default: 0.7)')
  parser.add_argument('-V', '--invalid_value', type=float, default=None,
                      help='Sample whose attribute score is equal to this '
                           'field will be ignored. (default: None)')

  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  logger = setup_logger(args.output_dir, logger_name='generate_data')

  logger.info('Loading latent codes.')
  if not os.path.isfile(args.latent_codes_path):
    raise ValueError(f'Latent codes `{args.latent_codes_path}` does not exist!')
  latent_codes = np.load(args.latent_codes_path)

  logger.info('Loading attribute scores.')
  if not os.path.isfile(args.scores_path):
    raise ValueError(f'Attribute scores `{args.scores_path}` does not exist!')
  scores = np.load(args.scores_path)

  boundary = train_boundary(latent_codes=latent_codes,
                            scores=scores,
                            chosen_num_or_ratio=args.chosen_num_or_ratio,
                            split_ratio=args.split_ratio,
                            invalid_value=args.invalid_value,
                            logger=logger)
  np.save(os.path.join(args.output_dir, 'boundary.npy'), boundary)


if __name__ == '__main__':
  main()
