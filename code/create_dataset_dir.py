"""Create new dataset directory structure
Generates new directory structure (as defined in README.md) under /datasets directory.
This is a convenience function which replicates the file structures used.

/datasets
    /example_dataset
      /data
      /eval
      /export
      /frames
          /eval
          /labeled
          /processed
          /raw
      /media
      /training

Note, must use file structure as defined in README.md
Arguments:
  [dataset_name]: name of new dataset to be created inside of datasets/ (required)
Example Usage:
  create_dataset_dir.py [dataset_name]
"""

import os
import sys
import argparse

# Define Arguments
parser = argparse.ArgumentParser(description='Create new dataset directory structure')
parser.add_argument('--dataset_name', help='name of new dataset to be created inside of datasets/ (required)', required=True)

args = parser.parse_args()
argsdict = vars(args)

# Parse Arguments
DATASET_NAME = argsdict['dataset_name']

PATH_TO_DIR = f'../datasets/{DATASET_NAME}'
RELATIVE_DIR_PATH = os.path.abspath(os.path.dirname(__file__) + '/' + PATH_TO_DIR)


if not os.path.exists(RELATIVE_DIR_PATH):
  
  # root
  os.makedirs(RELATIVE_DIR_PATH)

  # root/data
  DATA_PATH = f'{RELATIVE_DIR_PATH}/data'
  os.makedirs(DATA_PATH)

  # root/eval
  EVAL_PATH = f'{RELATIVE_DIR_PATH}/eval'
  os.makedirs(EVAL_PATH)

  # root/export
  EXPORT_PATH = f'{RELATIVE_DIR_PATH}/export'
  os.makedirs(EXPORT_PATH)

  # root/frames
  FRAMES_PATH = f'{RELATIVE_DIR_PATH}/frames'
  os.makedirs(FRAMES_PATH)

  # root/frames/eval
  FRAMES_EVAL_PATH = f'{FRAMES_PATH}/eval'
  os.makedirs(FRAMES_EVAL_PATH)

  # root/frames/labeled
  FRAMES_LABELED_PATH = f'{FRAMES_PATH}/labeled'
  os.makedirs(FRAMES_LABELED_PATH)

  # root/frames/processed
  FRAMES_PROCESSED_PATH = f'{FRAMES_PATH}/processed'
  os.makedirs(FRAMES_PROCESSED_PATH)

  # root/frames/raw
  FRAMES_RAW_PATH = f'{FRAMES_PATH}/raw'
  os.makedirs(FRAMES_RAW_PATH)

  # root/media
  MEDIA_PATH = f'{RELATIVE_DIR_PATH}/media'
  os.makedirs(MEDIA_PATH)

  # root/training
  TRAINING_PATH = f'{RELATIVE_DIR_PATH}/training'
  os.makedirs(TRAINING_PATH)

  # exit successful
  sys.exit(f'Dataset directory {DATASET_NAME} created successfully')

# exit error
sys.exit(f'Dataset with name {DATASET_NAME} already exists, please give a unique name')
